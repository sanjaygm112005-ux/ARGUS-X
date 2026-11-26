"""Tests for Milestone 9: Advanced Global Identity Management & Conflict Resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from association import AssociationConfig, CameraTransitionRule
from camera import CameraConfig, CameraManager
from identity_memory import (
    AuditLogEntry,
    IdentityState,
    MultiCameraIdentityManager,
)
from tracking.tracker import TrackedObject


def _unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _person(track_id: int) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(15.0, 25.0, 95.0, 225.0),
        class_id=0,
        class_name="person",
        confidence=0.93,
    )


class TestGlobalIdentityManagement(unittest.TestCase):
    """Tests for lifecycle persistence, conflict resolution, unresolved review, and audit logs."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.db_path = self.tmp_path / "global_identity.db"
        self.index_path = self.tmp_path / "global_identity.faiss"

        self.rules = [
            CameraTransitionRule("cam_1", "cam_1", min_travel_sec=0.0, max_travel_sec=300.0),
            CameraTransitionRule("cam_2", "cam_2", min_travel_sec=0.0, max_travel_sec=300.0),
            CameraTransitionRule("cam_1", "cam_2", min_travel_sec=5.0, max_travel_sec=120.0),
        ]
        self.cam_mgr = CameraManager(inactive_timeout_sec=0.5, extraction_interval_sec=0.0)
        self.cam_mgr.register_camera(CameraConfig(camera_id="cam_1"))
        self.cam_mgr.register_camera(CameraConfig(camera_id="cam_2"))
        self.cam_mgr.register_camera(CameraConfig(camera_id="cam_unknown"))
        self.cam_mgr.start_all()

        self.mc_mgr = MultiCameraIdentityManager(
            transition_rules=self.rules,
            config=AssociationConfig(max_time_gap_sec=300.0),
            expire_timeout_sec=100.0,
            db_path=self.db_path,
            index_path=self.index_path,
        )

    def tearDown(self) -> None:
        self.cam_mgr.close()
        self.mc_mgr.close()
        self._tmpdir.cleanup()

    def _step(
        self,
        detections_by_cam: dict[str, tuple[list[TrackedObject], dict[int, np.ndarray]]],
        timestamp: float,
    ):
        active = []
        for cam_id in self.cam_mgr.list_camera_ids():
            objs, embs = detections_by_cam.get(cam_id, ([], {}))
            res = self.cam_mgr.process_camera_detections(
                cam_id, objs, timestamp=timestamp, precomputed_embeddings=embs
            )
            active.extend(res.active_tracklets)
        return self.mc_mgr.process_tracklets(active, current_timestamp=timestamp)

    def test_identity_lifecycle_and_expiration_with_audit(self) -> None:
        v = _unit_vec(1)
        outs1 = self._step({"cam_1": ([_person(1)], {1: v})}, timestamp=10.0)
        gid = outs1[0].global_id
        assert gid is not None
        self.assertEqual(self.mc_mgr.get_identity(gid).state, IdentityState.ACTIVE)

        self._step({}, timestamp=12.0)
        self.assertEqual(self.mc_mgr.get_identity(gid).state, IdentityState.TEMPORARILY_LOST)

        outs2 = self._step({"cam_2": ([_person(2)], {2: v})}, timestamp=30.0)
        self.assertEqual(outs2[0].global_id, gid)
        self.assertEqual(self.mc_mgr.get_identity(gid).state, IdentityState.RECOVERED)

        # Lose and expire (> expire_timeout_sec=100.0)
        self._step({}, timestamp=32.0)
        self._step({}, timestamp=150.0)
        self.assertEqual(self.mc_mgr.get_identity(gid).state, IdentityState.EXPIRED)

        # Verify IdentityState.UNCERTAIN exists
        self.assertEqual(IdentityState.UNCERTAIN.value, "UNCERTAIN")

        # Verify audit log captures separate confidence components and decisions
        logs = self.mc_mgr.get_audit_logs(global_id=gid)
        decisions = [entry.decision for entry in logs]
        self.assertIn("NEW_IDENTITY", decisions)
        self.assertIn("RECOVERED_ACCEPT", decisions)
        self.assertIn("EXPIRED", decisions)
        rec_entry = next(e for e in logs if e.decision == "RECOVERED_ACCEPT")
        self.assertIsInstance(rec_entry, AuditLogEntry)
        self.assertIsNotNone(rec_entry.appearance_score)
        self.assertIsNotNone(rec_entry.temporal_score)
        self.assertIsNotNone(rec_entry.camera_score)
        self.assertIsNotNone(rec_entry.combined_confidence)

    def test_simultaneous_conflict_detection_and_duplicate_prevention(self) -> None:
        v = _unit_vec(2)
        outs = self._step(
            {
                "cam_1": ([_person(1)], {1: v}),
                "cam_2": ([_person(1)], {1: v}),
            },
            timestamp=10.0,
        )
        gid1 = outs[0].global_id
        gid2 = outs[1].global_id
        self.assertIsNotNone(gid1)
        self.assertIsNotNone(gid2)
        self.assertNotEqual(gid1, gid2)

        # Explicit conflict check when gid1 is active in cam_1
        conflict = self.mc_mgr.check_simultaneous_conflict(
            gid1, camera_id="cam_2", local_track_id=99, timestamp=10.5
        )
        self.assertTrue(conflict)
        self.assertEqual(len(self.mc_mgr.get_conflict_events()), 1)
        blocked_logs = self.mc_mgr.get_audit_logs(decision="CONFLICT_BLOCKED")
        self.assertEqual(len(blocked_logs), 1)

    def test_unknown_identity_management_and_manual_review(self) -> None:
        v = _unit_vec(3)
        outs1 = self._step({"cam_1": ([_person(1)], {1: v})}, timestamp=10.0)
        gid1 = outs1[0].global_id
        assert gid1 is not None
        self._step({}, timestamp=12.0)

        # Unknown transition cam_1 -> cam_unknown produces UNCERTAIN & PENDING record
        outs2 = self._step({"cam_unknown": ([_person(5)], {5: v})}, timestamp=25.0)
        self.assertIsNone(outs2[0].global_id)
        self.assertEqual(outs2[0].recovery_decision, "UNCERTAIN")

        pending = self.mc_mgr.list_unresolved_observations(status="PENDING")
        self.assertEqual(len(pending), 1)
        uid = pending[0].tracklet_uid
        self.assertEqual(pending[0].candidate_ids, [gid1])

        # Manual review resolves it to gid1
        resolved = self.mc_mgr.review_unresolved(
            tracklet_uid=uid,
            decision="ACCEPT",
            target_global_id=gid1,
            reason="operator_confirmed_corridor",
            timestamp=26.0,
        )
        self.assertEqual(resolved.status, "RESOLVED")
        self.assertEqual(resolved.resolved_global_id, gid1)
        self.assertEqual(self.mc_mgr.list_unresolved_observations(status="PENDING"), [])
        self.assertEqual(len(self.mc_mgr.get_audit_logs(decision="MANUAL_REVIEW_ACCEPT")), 1)

    def test_automatic_merge_forbidden_controlled_merge_audited(self) -> None:
        v1 = _unit_vec(4)
        v2 = _unit_vec(5)
        outs = self._step(
            {
                "cam_1": ([_person(1)], {1: v1}),
                "cam_2": ([_person(2)], {2: v2}),
            },
            timestamp=10.0,
        )
        g1, g2 = outs[0].global_id, outs[1].global_id
        assert g1 is not None and g2 is not None

        with self.assertRaises(ValueError):
            self.mc_mgr.merge_identities(g1, g2, reason="same jacket", operator_approved=False)

        self.mc_mgr.merge_identities(
            g1, g2, reason="operator_verified", operator_approved=True, timestamp=15.0
        )
        self.assertEqual(len(self.mc_mgr.get_audit_logs(decision="MANUAL_MERGE")), 1)

    def test_repeated_camera_disconnections_preserve_history(self) -> None:
        v = _unit_vec(6)
        outs = self._step({"cam_1": ([_person(1)], {1: v})}, timestamp=10.0)
        gid = outs[0].global_id
        assert gid is not None

        affected1 = self.mc_mgr.handle_camera_disconnect("cam_1", timestamp=11.0)
        affected2 = self.mc_mgr.handle_camera_disconnect("cam_1", timestamp=12.0)
        self.assertEqual(affected1, [gid])
        self.assertEqual(affected2, [])
        self.assertEqual(self.mc_mgr.get_identity(gid).state, IdentityState.TEMPORARILY_LOST)
        self.assertEqual(len(self.mc_mgr.memory.get_identity_observations(gid)), 1)
        self.assertEqual(len(self.mc_mgr.get_audit_logs(decision="CAMERA_DISCONNECTED")), 1)

    def test_persistence_and_application_restart_recovery(self) -> None:
        v = _unit_vec(7)
        outs1 = self._step({"cam_1": ([_person(1)], {1: v})}, timestamp=10.0)
        gid = outs1[0].global_id
        assert gid is not None

        self.mc_mgr.save_state(self.index_path)
        self.mc_mgr.close()

        # Simulate application restart with fresh CameraManager & MultiCameraIdentityManager
        restarted_cam = CameraManager(inactive_timeout_sec=0.5, extraction_interval_sec=0.0)
        restarted_cam.register_camera(CameraConfig(camera_id="cam_2"))
        restarted_cam.start_all()
        restarted_mgr = MultiCameraIdentityManager(
            transition_rules=self.rules,
            config=AssociationConfig(max_time_gap_sec=300.0),
            expire_timeout_sec=100.0,
            db_path=self.db_path,
            index_path=self.index_path,
        )
        try:
            restored = restarted_mgr.get_identity(gid)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.state, IdentityState.TEMPORARILY_LOST)
            self.assertGreaterEqual(len(restarted_mgr.get_audit_logs(global_id=gid)), 1)

            # Person appears in cam_2 after restart and recovers the persisted global_id
            res2 = restarted_cam.process_camera_detections(
                "cam_2", [_person(88)], timestamp=25.0, precomputed_embeddings={88: v}
            )
            outs2 = restarted_mgr.process_tracklets(res2.active_tracklets, current_timestamp=25.0)
            self.assertEqual(len(outs2), 1)
            self.assertEqual(outs2[0].global_id, gid)
            self.assertEqual(outs2[0].identity_state, IdentityState.RECOVERED)
        finally:
            restarted_cam.close()
            restarted_mgr.close()


if __name__ == "__main__":
    unittest.main()
