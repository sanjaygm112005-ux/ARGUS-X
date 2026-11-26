"""Tests for Milestone 10: ARGUS-X FastAPI Backend & Streaming Endpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app
from api.state import ArgusServiceState
from association import CameraTransitionRule
from tracking.tracker import TrackedObject


def _unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _person(track_id: int) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(20.0, 30.0, 100.0, 220.0),
        class_id=0,
        class_name="person",
        confidence=0.91,
    )


class TestArgusAPI(unittest.TestCase):
    """Integration tests for FastAPI REST and WebSocket endpoints."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self._tmpdir.name)
        rules = [
            CameraTransitionRule("cam_1", "cam_1", 0.0, 300.0),
            CameraTransitionRule("cam_1", "cam_2", 5.0, 120.0),
        ]
        self.service = ArgusServiceState(
            db_path=tmp / "api_test.db",
            index_path=tmp / "api_test.faiss",
            transition_rules=rules,
        )
        self.app = create_app(service_state=self.service)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.service.close()
        self._tmpdir.cleanup()

    def _push_detection(self, cam_id: str, track_id: int, vec: np.ndarray, ts: float):
        res = self.service.camera_manager.process_camera_detections(
            cam_id,
            [_person(track_id)],
            timestamp=ts,
            precomputed_embeddings={track_id: vec},
        )
        all_active = []
        for cid in self.service.camera_manager.list_camera_ids():
            all_active.extend(
                self.service.camera_manager.get_camera(cid).tracklet_manager.get_active_tracklets()
            )
        outs = self.service.identity_manager.process_tracklets(all_active, current_timestamp=ts)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cam_outs = [o for o in outs if o.camera_id == cam_id]
        self.service.ingest_frame_outputs(cam_id, frame, cam_outs, ts)
        return outs

    def test_health_and_statistics_endpoints(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "ARGUS-X")

        stats = self.client.get("/statistics")
        self.assertEqual(stats.status_code, 200)
        self.assertIn("total_cameras", stats.json())
        self.assertIn("active_identities", stats.json())

    def test_camera_registration_listing_frame_ws_and_disconnection(self) -> None:
        reg = self.client.post(
            "/cameras/register",
            json={"camera_id": "cam_1", "name": "Lobby Cam", "auto_start": True},
        )
        self.assertEqual(reg.status_code, 201)
        self.assertTrue(reg.json()["camera"]["is_running"])

        listing = self.client.get("/cameras")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["count"], 1)

        v = _unit_vec(1)
        outs = self._push_detection("cam_1", 1, v, ts=10.0)
        gid = outs[0].global_id
        assert gid is not None

        frame_resp = self.client.get("/cameras/cam_1/frame")
        self.assertEqual(frame_resp.status_code, 200)
        self.assertEqual(frame_resp.headers["content-type"], "image/jpeg")

        with self.client.websocket_connect("/ws/cameras/cam_1") as ws:
            payload = ws.receive_json()
            self.assertEqual(payload["camera_id"], "cam_1")
            self.assertTrue(payload["is_running"])
            self.assertIsNotNone(payload["frame_b64"])

        disc = self.client.post("/cameras/cam_1/disconnect")
        self.assertEqual(disc.status_code, 200)
        self.assertEqual(disc.json()["affected_global_ids"], [gid])
        self.assertFalse(disc.json()["is_running"])

    def test_identities_unresolved_manual_review_and_audit(self) -> None:
        self.client.post(
            "/cameras/register", json={"camera_id": "cam_1", "auto_start": True}
        )
        self.client.post(
            "/cameras/register", json={"camera_id": "cam_unknown", "auto_start": True}
        )

        v = _unit_vec(2)
        outs1 = self._push_detection("cam_1", 1, v, ts=10.0)
        gid = outs1[0].global_id
        assert gid is not None

        # Disconnect cam_1 -> identity becomes TEMPORARILY_LOST
        self.client.post("/cameras/cam_1/disconnect")

        # Unknown transition to cam_unknown -> UNCERTAIN / PENDING
        self._push_detection("cam_unknown", 5, v, ts=25.0)

        identities = self.client.get("/identities").json()
        self.assertEqual(identities["count"], 1)

        details = self.client.get(f"/identities/{gid}")
        self.assertEqual(details.status_code, 200)
        self.assertEqual(details.json()["identity"]["global_id"], gid)
        self.assertGreaterEqual(len(details.json()["observations"]), 1)

        unresolved = self.client.get("/unresolved").json()
        self.assertEqual(unresolved["count"], 1)
        obs_id = unresolved["unresolved"][0]["observation_id"]

        # Unconfirmed review must be rejected (400)
        unconfirmed = self.client.post(
            f"/unresolved/{obs_id}/review",
            json={
                "decision": "ACCEPT",
                "target_global_id": gid,
                "reason": "test",
                "operator_confirmed": False,
            },
        )
        self.assertEqual(unconfirmed.status_code, 400)

        # Confirmed review succeeds
        confirmed = self.client.post(
            f"/unresolved/{obs_id}/review",
            json={
                "decision": "ACCEPT",
                "target_global_id": gid,
                "reason": "operator_verified",
                "operator_confirmed": True,
            },
        )
        self.assertEqual(confirmed.status_code, 200)
        self.assertEqual(confirmed.json()["status"], "RESOLVED")
        self.assertEqual(confirmed.json()["resolved_global_id"], gid)

        audit = self.client.get("/audit")
        self.assertEqual(audit.status_code, 200)
        decisions = [e["decision"] for e in audit.json()["audit_logs"]]
        self.assertIn("MANUAL_REVIEW_ACCEPT", decisions)

    def test_invalid_requests_return_proper_errors(self) -> None:
        self.assertEqual(self.client.get("/identities/nonexistent").status_code, 404)
        self.assertEqual(
            self.client.post("/cameras/missing_cam/disconnect").status_code, 404
        )
        self.assertEqual(
            self.client.post(
                "/cameras/register", json={"camera_id": ""}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/unresolved/missing_obs/review",
                json={
                    "decision": "REJECT",
                    "reason": "not_matching",
                    "operator_confirmed": True,
                },
            ).status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
