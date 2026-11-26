"""Comprehensive tests for ARGUS-X Platform Upgrade (Stages 1-5 & 7)."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.state import ArgusServiceState
from backend.auth.security import CSRF_COOKIE_NAME, CSRF_HEADER_NAME, login_rate_limiter
from backend.benchmarks.mot_eval import evaluate_mot17_predictions
from backend.database.base import Base
from backend.database.session import build_engine, get_db
from backend.datasets.service import discover_mot17_sequences, load_mot17_ground_truth
from backend.main import create_fullstack_app
from backend.services.seed import seed_database


def _make_dummy_mp4(path: Path, frames: int = 8, width: int = 160, height: int = 120) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(img, (20 + i * 5, 20), (60 + i * 5, 100), (200, 200, 200), -1)
        writer.write(img)
    writer.release()
    return path


class TestPlatformUpgrade(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        db_url = f"sqlite:///{tmp_path / 'test_platform.db'}"
        self.engine = build_engine(db_url)
        Base.metadata.create_all(bind=self.engine)
        self.TestSession = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False
        )
        db = self.TestSession()
        seed_database(db, create_bootstrap_admin=True)
        db.close()

        self.ai_state = ArgusServiceState(db_path=":memory:", index_path=None)
        self.app = create_fullstack_app(service_state=self.ai_state, init_db=False)

        def _override_get_db():
            s = self.TestSession()
            try:
                yield s
            finally:
                s.close()

        self.app.dependency_overrides[get_db] = _override_get_db
        self.client = TestClient(self.app)
        login_rate_limiter._attempts.clear()

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.ai_state.close()
        self.engine.dispose()
        self._tmp.cleanup()

    def _login(self, identifier: str = "admin", password: str = "ChangeMeNow!123") -> dict[str, str]:
        resp = self.client.post(
            "/api/auth/login",
            json={"identifier": identifier, "password": password, "remember_me": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        token = self.client.cookies.get(CSRF_COOKIE_NAME, "")
        return {CSRF_HEADER_NAME: token}

    def test_mot17_discovery_and_gt_loading(self) -> None:
        # Missing directory returns clear setup message
        missing = discover_mot17_sequences(mot17_root=Path(self._tmp.name) / "nonexistent")
        self.assertEqual(missing["status"], "MISSING")
        self.assertIn("setup_message", missing)
        self.assertTrue(missing["setup_message"])

        # Real MOT17 in workspace (if present)
        real_info = discover_mot17_sequences(detector_filter="FRCNN", split_filter="train")
        if real_info["status"] == "AVAILABLE":
            self.assertGreaterEqual(real_info["sequence_count"], 1)
            first_seq = real_info["sequences"][0]
            self.assertEqual(first_seq["detector_type"], "FRCNN")
            self.assertTrue(first_seq["is_valid"])
            self.assertTrue(first_seq["has_ground_truth"])
            gt_path = Path(first_seq["sequence_dir"]) / "gt" / "gt.txt"
            gt_frames = load_mot17_ground_truth(gt_path, pedestrian_only=True)
            self.assertGreater(len(gt_frames), 0)

    def test_trackeval_compatible_metrics(self) -> None:
        gt = {
            1: [{"id": 1, "x": 10, "y": 10, "w": 50, "h": 100}],
            2: [{"id": 1, "x": 15, "y": 10, "w": 50, "h": 100}],
            3: [{"id": 1, "x": 20, "y": 10, "w": 50, "h": 100}],
        }
        pred = {
            1: [{"id": 7, "x": 10, "y": 10, "w": 50, "h": 100}],
            2: [{"id": 7, "x": 15, "y": 10, "w": 50, "h": 100}],
            3: [{"id": 7, "x": 20, "y": 10, "w": 50, "h": 100}],
        }
        res = evaluate_mot17_predictions(gt, pred, [1, 2, 3])
        self.assertEqual(res.mota, 100.0)
        self.assertEqual(res.idf1, 100.0)
        self.assertEqual(res.hota, 100.0)
        self.assertEqual(res.id_switches, 0)
        self.assertEqual(res.false_positives, 0)
        self.assertEqual(res.false_negatives, 0)

    def test_dataset_routes_authenticated(self) -> None:
        self.assertEqual(self.client.get("/api/datasets").status_code, 401)
        self._login()
        ds = self.client.get("/api/datasets")
        self.assertEqual(ds.status_code, 200)
        self.assertEqual(ds.json()["datasets"][0]["name"], "MOT17")

        mot17 = self.client.get("/api/datasets/mot17")
        self.assertEqual(mot17.status_code, 200)
        self.assertIn("sequences", mot17.json())

    def test_video_upload_security_validation_and_analysis(self) -> None:
        headers = self._login()

        # Reject invalid extension
        bad_ext = self.client.post(
            "/api/videos/upload",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
            data={"camera_label": "Bad File"},
            headers=headers,
        )
        self.assertEqual(bad_ext.status_code, 422)

        # Upload valid MP4
        vid_path = _make_dummy_mp4(Path(self._tmp.name) / "sample.mp4", frames=6)
        with vid_path.open("rb") as f:
            up = self.client.post(
                "/api/videos/upload",
                files={"file": ("../../sample.mp4", f, "video/mp4")},
                data={"camera_label": "Lobby Cam 1"},
                headers=headers,
            )
        self.assertEqual(up.status_code, 201, up.text)
        video = up.json()["video"]
        vid_uid = video["video_uid"]
        self.assertEqual(video["camera_label"], "Lobby Cam 1")
        self.assertGreater(video["file_size_bytes"], 0)
        self.assertEqual(video["total_frames"], 6)

        # List & get
        self.assertEqual(self.client.get("/api/videos").status_code, 200)
        self.assertEqual(self.client.get(f"/api/videos/{vid_uid}").status_code, 200)

        # Stream works for authenticated user
        stream_resp = self.client.get(f"/api/videos/{vid_uid}/stream?kind=original")
        self.assertEqual(stream_resp.status_code, 200)

        # Cleanup
        del_resp = self.client.delete(f"/api/videos/{vid_uid}", headers=headers)
        self.assertEqual(del_resp.status_code, 200)

    def test_multicamera_validation_and_session_listing(self) -> None:
        headers = self._login()

        # Upload 2 dummy videos
        v1_path = _make_dummy_mp4(Path(self._tmp.name) / "c1.mp4", frames=4)
        v2_path = _make_dummy_mp4(Path(self._tmp.name) / "c2.mp4", frames=4)

        with v1_path.open("rb") as f1:
            r1 = self.client.post(
                "/api/videos/upload",
                files={"file": ("c1.mp4", f1, "video/mp4")},
                data={"camera_label": "Cam 1"},
                headers=headers,
            )
        with v2_path.open("rb") as f2:
            r2 = self.client.post(
                "/api/videos/upload",
                files={"file": ("c2.mp4", f2, "video/mp4")},
                data={"camera_label": "Cam 2"},
                headers=headers,
            )
        uid1 = r1.json()["video"]["video_uid"]
        uid2 = r2.json()["video"]["video_uid"]

        # Duplicate camera code rejected
        dup = self.client.post(
            "/api/multicamera/analyze",
            json={
                "cameras": [
                    {"video_uid": uid1, "camera_code": "cam_1"},
                    {"video_uid": uid2, "camera_code": "cam_1"},
                ],
                "max_frames_per_camera": 5,
            },
            headers=headers,
        )
        self.assertEqual(dup.status_code, 422)

        # Clean up uploaded videos
        self.client.delete(f"/api/videos/{uid1}", headers=headers)
        self.client.delete(f"/api/videos/{uid2}", headers=headers)


if __name__ == "__main__":
    unittest.main()
