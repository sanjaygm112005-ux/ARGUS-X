"""Tests for Milestone 8: Multi-Camera Long-Term Identity Recovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from association import AssociationConfig, CameraTransitionRule
from camera import CameraConfig, CameraManager
from identity_memory import (
    IdentityState,
    MultiCameraIdentityManager,
    MultiCameraRecoveryOutput,
)
from tracking.tracker import TrackedObject


def _unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _person(track_id: int, bbox: tuple[float, float, float, float] = (20, 30, 100, 230)) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=bbox,
        class_id=0,
        class_name="person",
        confidence=0.90,
    )


class TestMultiCameraIdentityRecovery(unittest.TestCase):
    """Controlled synthetic tests for multi-camera identity recovery and CameraManager."""

    def setUp(self) -> None:
        self.rules = [
            CameraTransitionRule("cam_1", "cam_1", min_travel_sec=0.0, max_travel_sec=600.0),
            CameraTransitionRule("cam_2", "cam_2", min_travel_sec=0.0, max_travel_sec=600.0),
            CameraTransitionRule("cam_3", "cam_3", min_travel_sec=0.0, max_travel_sec=600.0),
            CameraTransitionRule("cam_1", "cam_2", min_travel_sec=5.0, max_travel_sec=120.0),
            CameraTransitionRule("cam_2", "cam_3", min_travel_sec=5.0, max_travel_sec=120.0),
            CameraTransitionRule("cam_1", "cam_3", allowed=False),
        ]
        self.cam_mgr = CameraManager(inactive_timeout_sec=0.5, extraction_interval_sec=0.0)
        self.cam_mgr.register_camera(CameraConfig(camera_id="cam_1"))
        self.cam_mgr.register_camera(CameraConfig(camera_id="cam_2"))
        self.cam_mgr.register_camera(CameraConfig(camera_id="cam_3"))
        self.cam_mgr.start_all()

        self.mc_mgr = MultiCameraIdentityManager(
            transition_rules=self.rules,
            config=AssociationConfig(max_time_gap_sec=600.0),
            expire_timeout_sec=600.0,
        )

    def tearDown(self) -> None:
        self.cam_mgr.close()
        self.mc_mgr.close()

    def _step(
        self,
        detections_by_cam: dict[str, tuple[list[TrackedObject], dict[int, np.ndarray]]],
        timestamp: float,
    ) -> list[MultiCameraRecoveryOutput]:
        all_active = []
        for cam_id in self.cam_mgr.list_camera_ids():
            objs, embs = detections_by_cam.get(cam_id, ([], {}))
            res = self.cam_mgr.process_camera_detections(
                cam_id, objs, timestamp=timestamp, precomputed_embeddings=embs
            )
            all_active.extend(res.active_tracklets)
        return self.mc_mgr.process_tracklets(all_active, current_timestamp=timestamp)

    def test_same_person_cam1_to_cam2_to_cam3(self) -> None:
        v = _unit_vec(1)
        # Appears in cam_1 at t=10
        outs1 = self._step({"cam_1": ([_person(1)], {1: v})}, timestamp=10.0)
        self.assertEqual(len(outs1), 1)
        gid = outs1[0].global_id
        assert gid is not None
        self.assertEqual(outs1[0].identity_state, IdentityState.ACTIVE)

        # Leaves cam_1 at t=11
        self._step({}, timestamp=11.0)

        # Appears in cam_2 at t=25 with different local_id=42
        outs2 = self._step({"cam_2": ([_person(42)], {42: v})}, timestamp=25.0)
        self.assertEqual(len(outs2), 1)
        o2 = outs2[0]
        self.assertEqual(o2.global_id, gid)
        self.assertEqual(o2.camera_id, "cam_2")
        self.assertEqual(o2.local_track_id, 42)
        self.assertEqual(o2.identity_state, IdentityState.RECOVERED)
        self.assertEqual(o2.recovery_decision, "RECOVERED_ACCEPT")
        self.assertEqual(o2.previous_camera_id, "cam_1")
        self.assertAlmostEqual(o2.previous_observation_timestamp, 10.0)

        # Leaves cam_2 at t=26
        self._step({}, timestamp=26.0)

        # Appears in cam_3 at t=40 with local_id=7
        outs3 = self._step({"cam_3": ([_person(7)], {7: v})}, timestamp=40.0)
        o3 = outs3[0]
        self.assertEqual(o3.global_id, gid)
        self.assertEqual(o3.previous_camera_id, "cam_2")
        self.assertAlmostEqual(o3.previous_observation_timestamp, 25.0)

        rec = self.mc_mgr.get_identity(gid)
        assert rec is not None
        self.assertEqual(rec.camera_sequence, ["cam_1", "cam_2", "cam_3"])
        self.assertEqual(rec.camera_track_history, {"cam_1": [1], "cam_2": [42], "cam_3": [7]})
        # History preserved in shared FAISS/SQLite memory
        self.assertEqual(len(self.mc_mgr.memory.get_identity_observations(gid)), 3)

    def test_different_people_and_identical_local_ids_across_cameras(self) -> None:
        v1 = _unit_vec(10)
        v2 = -v1

        outs1 = self._step({"cam_1": ([_person(1)], {1: v1})}, timestamp=10.0)
        gid1 = outs1[0].global_id
        self._step({}, timestamp=11.0)

        # Different person in cam_2 also has local_track_id=1
        outs2 = self._step({"cam_2": ([_person(1)], {1: v2})}, timestamp=20.0)
        gid2 = outs2[0].global_id
        self.assertIsNotNone(gid2)
        self.assertNotEqual(gid1, gid2)

    def test_impossible_camera_transition_and_invalid_travel_times(self) -> None:
        v = _unit_vec(20)
        outs1 = self._step({"cam_1": ([_person(1)], {1: v})}, timestamp=10.0)
        gid1 = outs1[0].global_id
        self._step({}, timestamp=11.0)

        # Forbidden direct cam_1 -> cam_3
        outs_forb = self._step({"cam_3": ([_person(2)], {2: v})}, timestamp=30.0)
        self.assertNotEqual(outs_forb[0].global_id, gid1)

        # Too fast cam_1 -> cam_2 (dt=1s < min_travel_sec=5s)
        v_other = _unit_vec(21)
        outs_a = self._step({"cam_1": ([_person(3)], {3: v_other})}, timestamp=50.0)
        gid_a = outs_a[0].global_id
        self._step({}, timestamp=50.6)
        outs_fast = self._step({"cam_2": ([_person(4)], {4: v_other})}, timestamp=51.0)
        self.assertNotEqual(outs_fast[0].global_id, gid_a)

    def test_unknown_camera_transition_remains_uncertain(self) -> None:
        self.cam_mgr.register_camera(CameraConfig(camera_id="cam_unknown"))
        self.cam_mgr.start_camera("cam_unknown")

        v = _unit_vec(30)
        self._step({"cam_1": ([_person(1)], {1: v})}, timestamp=10.0)
        self._step({}, timestamp=11.0)

        outs = self._step({"cam_unknown": ([_person(5)], {5: v})}, timestamp=25.0)
        self.assertEqual(len(outs), 1)
        self.assertIsNone(outs[0].global_id)
        self.assertEqual(outs[0].identity_state, IdentityState.UNKNOWN)
        self.assertEqual(outs[0].recovery_decision, "UNCERTAIN")
        self.assertEqual(outs[0].previous_camera_id, "cam_1")

    def test_simultaneous_appearances_prevent_duplicate_global_id(self) -> None:
        v = _unit_vec(40)
        # Person is currently ACTIVE in cam_1, and a similar-looking person appears
        # simultaneously in cam_2 at t=10.0 -> must NOT share the same global_id.
        outs = self._step(
            {
                "cam_1": ([_person(1)], {1: v}),
                "cam_2": ([_person(1)], {1: v}),
            },
            timestamp=10.0,
        )
        self.assertEqual(len(outs), 2)
        self.assertIsNotNone(outs[0].global_id)
        self.assertIsNotNone(outs[1].global_id)
        self.assertNotEqual(outs[0].global_id, outs[1].global_id)

    def test_multiple_people_and_similar_looking_moving_between_cameras(self) -> None:
        base = _unit_vec(50)
        person_a = base + 0.08 * _unit_vec(51)
        person_a /= np.linalg.norm(person_a)
        person_b = base + 0.08 * _unit_vec(52)
        person_b /= np.linalg.norm(person_b)

        outs1 = self._step(
            {"cam_1": ([_person(1), _person(2)], {1: person_a, 2: person_b})},
            timestamp=10.0,
        )
        gid_by_old = {o.local_track_id: o.global_id for o in outs1}
        self._step({}, timestamp=11.0)

        # Both appear in cam_2 with swapped local IDs 10 and 11
        outs2 = self._step(
            {"cam_2": ([_person(10), _person(11)], {10: person_b, 11: person_a})},
            timestamp=25.0,
        )
        gid_by_new = {o.local_track_id: o.global_id for o in outs2}
        self.assertEqual(gid_by_new[10], gid_by_old[2])
        self.assertEqual(gid_by_new[11], gid_by_old[1])
        self.assertEqual(len(self.mc_mgr.list_identities()), 2)

    def test_camera_manager_independent_trackers_and_video_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vid1 = Path(tmpdir) / "cam1.mp4"
            vid2 = Path(tmpdir) / "cam2.mp4"
            for vp in (vid1, vid2):
                writer = cv2.VideoWriter(
                    str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
                )
                for _ in range(3):
                    writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
                writer.release()

            mgr = CameraManager()
            try:
                mgr.register_camera(CameraConfig(camera_id="v1", source=vid1))
                mgr.register_camera(CameraConfig(camera_id="v2", source=vid2))
                mgr.start_all()

                s1 = mgr.get_camera("v1")
                s2 = mgr.get_camera("v2")
                self.assertIsNot(s1.tracklet_manager, s2.tracklet_manager)
                self.assertTrue(s1.is_running and s2.is_running)
                self.assertIsNotNone(mgr.read_frame("v1"))
                self.assertIsNotNone(mgr.read_frame("v2"))
            finally:
                mgr.close()

            self.assertFalse(mgr.get_camera("v1").is_running)
            self.assertIsNone(mgr.get_camera("v1").capture)
            self.assertFalse(mgr.get_camera("v2").is_running)
            self.assertIsNone(mgr.get_camera("v2").capture)


if __name__ == "__main__":
    unittest.main()
