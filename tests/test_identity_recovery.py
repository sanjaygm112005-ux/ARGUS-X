"""Tests for Milestone 7: Single-Camera Long-Term Identity Recovery."""

from __future__ import annotations

import unittest

import numpy as np

from association import AssociationConfig, CameraTransitionRule, DataAssociator
from identity_memory import (
    GlobalIdentityManager,
    IdentityMemory,
    IdentityState,
    RecoveryOutput,
    TrackletManager,
)
from tracking.tracker import TrackedObject


def _unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _person(track_id: int, bbox: tuple[float, float, float, float] = (10, 20, 90, 220)) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=bbox,
        class_id=0,
        class_name="person",
        confidence=0.92,
    )


class TestSingleCameraIdentityRecovery(unittest.TestCase):
    """Controlled synthetic tests for single-camera identity recovery."""

    def setUp(self) -> None:
        self.tm = TrackletManager(
            camera_id="cam_0",
            inactive_timeout_sec=0.5,
            extraction_interval_sec=0.0,
        )
        self.gm = GlobalIdentityManager(
            camera_id="cam_0",
            lost_timeout_sec=10.0,
            expire_timeout_sec=100.0,
        )

    def tearDown(self) -> None:
        self.gm.close()

    def test_new_identity_creation_and_visualization_output(self) -> None:
        v = _unit_vec(1)
        upd = self.tm.update([_person(1)], timestamp=1.0, precomputed_embeddings={1: v})
        outs = self.gm.process_tracklets(upd.active_tracklets, current_timestamp=1.0)

        self.assertEqual(len(outs), 1)
        out = outs[0]
        self.assertIsInstance(out, RecoveryOutput)
        self.assertEqual(out.camera_id, "cam_0")
        self.assertEqual(out.local_track_id, 1)
        self.assertIsNotNone(out.global_id)
        self.assertEqual(out.bbox, (10.0, 20.0, 90.0, 220.0))
        self.assertEqual(out.identity_state, IdentityState.ACTIVE)
        self.assertEqual(out.recovery_decision, "NEW_IDENTITY")
        d = out.to_dict()
        self.assertEqual(d["identity_state"], "ACTIVE")

    def test_same_person_recovery_after_temporary_occlusion(self) -> None:
        v = _unit_vec(2)
        # t=1.0: person appears with local_id=1
        upd1 = self.tm.update([_person(1)], timestamp=1.0, precomputed_embeddings={1: v})
        outs1 = self.gm.process_tracklets(upd1.active_tracklets, current_timestamp=1.0)
        orig_gid = outs1[0].global_id
        assert orig_gid is not None

        # t=2.0: person disappears (occluded)
        upd2 = self.tm.update([], timestamp=2.0)
        self.gm.process_tracklets(upd2.active_tracklets, current_timestamp=2.0)
        self.assertEqual(self.gm.get_identity(orig_gid).state, IdentityState.TEMPORARILY_LOST)

        # t=5.0: same person reappears with new local_id=9
        upd3 = self.tm.update([_person(9)], timestamp=5.0, precomputed_embeddings={9: v})
        outs3 = self.gm.process_tracklets(upd3.active_tracklets, current_timestamp=5.0)
        self.assertEqual(len(outs3), 1)
        self.assertEqual(outs3[0].global_id, orig_gid)
        self.assertEqual(outs3[0].local_track_id, 9)
        self.assertEqual(outs3[0].identity_state, IdentityState.RECOVERED)
        self.assertEqual(outs3[0].recovery_decision, "RECOVERED_ACCEPT")
        self.assertEqual(self.gm.get_identity(orig_gid).local_track_history, [1, 9])

    def test_different_person_rejected_creates_new_identity(self) -> None:
        v1 = _unit_vec(10)
        v2 = -v1

        upd1 = self.tm.update([_person(1)], timestamp=1.0, precomputed_embeddings={1: v1})
        gid1 = self.gm.process_tracklets(upd1.active_tracklets, current_timestamp=1.0)[0].global_id

        upd2 = self.tm.update([], timestamp=2.0)
        self.gm.process_tracklets(upd2.active_tracklets, current_timestamp=2.0)

        upd3 = self.tm.update([_person(2)], timestamp=4.0, precomputed_embeddings={2: v2})
        outs3 = self.gm.process_tracklets(upd3.active_tracklets, current_timestamp=4.0)
        gid2 = outs3[0].global_id

        self.assertIsNotNone(gid2)
        self.assertNotEqual(gid1, gid2)
        self.assertEqual(outs3[0].identity_state, IdentityState.ACTIVE)
        self.assertEqual(len(self.gm.list_identities()), 2)

    def test_long_disappearance_and_expired_identity_preserves_history(self) -> None:
        v = _unit_vec(20)
        upd1 = self.tm.update([_person(1)], timestamp=10.0, precomputed_embeddings={1: v})
        gid1 = self.gm.process_tracklets(upd1.active_tracklets, current_timestamp=10.0)[0].global_id
        assert gid1 is not None

        # Disappear at t=11.0
        upd2 = self.tm.update([], timestamp=11.0)
        self.gm.process_tracklets(upd2.active_tracklets, current_timestamp=11.0)

        # Reappear at t=200.0 (> expire_timeout_sec=100.0)
        upd3 = self.tm.update([_person(7)], timestamp=200.0, precomputed_embeddings={7: v})
        outs3 = self.gm.process_tracklets(upd3.active_tracklets, current_timestamp=200.0)

        self.assertEqual(self.gm.get_identity(gid1).state, IdentityState.EXPIRED)
        self.assertNotEqual(outs3[0].global_id, gid1)
        # Historical observations for expired identity are preserved.
        self.assertEqual(len(self.gm.memory.get_identity_observations(gid1)), 1)

    def test_multiple_people_reappearing_simultaneously(self) -> None:
        va = _unit_vec(30)
        vb = _unit_vec(31)

        upd1 = self.tm.update(
            [_person(1), _person(2)],
            timestamp=1.0,
            precomputed_embeddings={1: va, 2: vb},
        )
        outs1 = self.gm.process_tracklets(upd1.active_tracklets, current_timestamp=1.0)
        gid_by_old = {o.local_track_id: o.global_id for o in outs1}

        upd2 = self.tm.update([], timestamp=2.0)
        self.gm.process_tracklets(upd2.active_tracklets, current_timestamp=2.0)

        # Both reappear simultaneously with swapped/new local IDs 11 and 12.
        upd3 = self.tm.update(
            [_person(11), _person(12)],
            timestamp=5.0,
            precomputed_embeddings={11: vb, 12: va},
        )
        outs3 = self.gm.process_tracklets(upd3.active_tracklets, current_timestamp=5.0)
        gid_by_new = {o.local_track_id: o.global_id for o in outs3}

        self.assertEqual(gid_by_new[11], gid_by_old[2])
        self.assertEqual(gid_by_new[12], gid_by_old[1])
        self.assertTrue(all(o.identity_state == IdentityState.RECOVERED for o in outs3))

    def test_similar_looking_people_never_merged(self) -> None:
        base = _unit_vec(40)
        twin_a = base + 0.05 * _unit_vec(41)
        twin_a /= np.linalg.norm(twin_a)
        twin_b = base + 0.05 * _unit_vec(42)
        twin_b /= np.linalg.norm(twin_b)

        upd1 = self.tm.update(
            [_person(1), _person(2)],
            timestamp=1.0,
            precomputed_embeddings={1: twin_a, 2: twin_b},
        )
        outs1 = self.gm.process_tracklets(upd1.active_tracklets, current_timestamp=1.0)
        self.assertNotEqual(outs1[0].global_id, outs1[1].global_id)
        self.assertEqual(len(self.gm.list_identities()), 2)

    def test_uncertain_association_stays_unresolved(self) -> None:
        v1 = _unit_vec(50)
        ortho = _unit_vec(51)
        ortho = ortho - np.dot(ortho, v1) * v1
        ortho /= np.linalg.norm(ortho)
        # Construct vector with ~0.50 cosine similarity (borderline/uncertain).
        borderline = 0.50 * v1 + np.sqrt(1.0 - 0.25) * ortho
        borderline /= np.linalg.norm(borderline)

        memory = IdentityMemory(db_path=":memory:")
        config = AssociationConfig(
            max_time_gap_sec=100.0,
            min_appearance_similarity=0.35,
            accept_threshold=0.80,
            uncertain_threshold=0.50,
        )
        associator = DataAssociator(
            memory=memory,
            config=config,
            transition_rules=[CameraTransitionRule("cam_0", "cam_0", 0.0, 100.0)],
        )
        gm = GlobalIdentityManager(
            camera_id="cam_0",
            memory=memory,
            associator=associator,
            expire_timeout_sec=100.0,
        )
        try:
            upd1 = self.tm.update([_person(1)], timestamp=1.0, precomputed_embeddings={1: v1})
            gm.process_tracklets(upd1.active_tracklets, current_timestamp=1.0)

            upd2 = self.tm.update([], timestamp=2.0)
            gm.process_tracklets(upd2.active_tracklets, current_timestamp=2.0)

            upd3 = self.tm.update([_person(2)], timestamp=10.0, precomputed_embeddings={2: borderline})
            outs3 = gm.process_tracklets(upd3.active_tracklets, current_timestamp=10.0)
            self.assertEqual(outs3[0].recovery_decision, "UNCERTAIN")
            self.assertEqual(outs3[0].identity_state, IdentityState.UNKNOWN)
            self.assertIsNone(outs3[0].global_id)
        finally:
            gm.close()
            memory.close()

    def test_duplicate_local_ids_across_separate_tracklets(self) -> None:
        v1 = _unit_vec(60)
        v2 = -v1  # completely different person later given the same local_id=1

        upd1 = self.tm.update([_person(1)], timestamp=1.0, precomputed_embeddings={1: v1})
        gid1 = self.gm.process_tracklets(upd1.active_tracklets, current_timestamp=1.0)[0].global_id

        upd2 = self.tm.update([], timestamp=2.0)
        self.gm.process_tracklets(upd2.active_tracklets, current_timestamp=2.0)

        # Different person appears later with reused local_track_id=1.
        upd3 = self.tm.update([_person(1)], timestamp=4.0, precomputed_embeddings={1: v2})
        outs3 = self.gm.process_tracklets(upd3.active_tracklets, current_timestamp=4.0)
        gid2 = outs3[0].global_id

        self.assertIsNotNone(gid2)
        self.assertNotEqual(gid1, gid2)
        self.assertEqual(len(self.gm.list_identities()), 2)

    def test_single_camera_constraint_enforced(self) -> None:
        other_tm = TrackletManager(camera_id="cam_other", inactive_timeout_sec=0.5)
        upd = other_tm.update([_person(1)], timestamp=1.0, precomputed_embeddings={1: _unit_vec(70)})
        with self.assertRaises(ValueError):
            self.gm.process_tracklets(upd.active_tracklets, current_timestamp=1.0)


if __name__ == "__main__":
    unittest.main()
