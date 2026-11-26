"""Tests for Milestone 6: Confidence-Aware Data Association (Hungarian Algorithm)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from association import (
    GATING_COST,
    AssociationConfig,
    CameraTransitionRule,
    ConfidenceDecision,
    CostMatrixBuilder,
    DataAssociator,
    IdentityCandidate,
    KalmanMotionEstimate,
    TrackletObservation,
)
from identity_memory import IdentityMemory


def _unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestAssociation(unittest.TestCase):
    """Unit and integration tests for Hungarian data association."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "assoc_test.db"
        self.memory = IdentityMemory(embedding_dim=512, db_path=self.db_path)

    def tearDown(self) -> None:
        self.memory.close()
        self._tmpdir.cleanup()

    def test_synthetic_cost_matrix_hungarian_one_to_one(self) -> None:
        associator = DataAssociator()
        # Greedy would pick (0,0)=0.1, leaving (1,1)=0.9 (total 1.0).
        # Hungarian picks (0,1)=0.2 and (1,0)=0.2 (total 0.4).
        cost = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.9],
            ],
            dtype=np.float64,
        )
        pairs, unmatched_r, unmatched_c = associator.solve_assignment(cost)
        self.assertEqual(sorted(pairs), [(0, 1), (1, 0)])
        self.assertEqual(unmatched_r, [])
        self.assertEqual(unmatched_c, [])

    def test_synthetic_rectangular_cost_matrix_with_gating(self) -> None:
        associator = DataAssociator()
        cost = np.array(
            [
                [0.15, GATING_COST],
                [0.80, 0.25],
                [GATING_COST, GATING_COST],
            ],
            dtype=np.float64,
        )
        pairs, unmatched_r, unmatched_c = associator.solve_assignment(cost)
        self.assertEqual(sorted(pairs), [(0, 0), (1, 1)])
        self.assertEqual(unmatched_r, [2])
        self.assertEqual(unmatched_c, [])

    def test_invalid_cost_matrices_raise_value_error(self) -> None:
        associator = DataAssociator()
        invalid_matrices = [
            None,
            np.array([0.1, 0.2]),
            np.array([[0.1, np.nan], [0.2, 0.3]]),
            np.array([[0.1, np.inf], [0.2, 0.3]]),
            np.array([[0.1, -0.5], [0.2, 0.3]]),
        ]
        for bad in invalid_matrices:
            with self.assertRaises(ValueError):
                associator.solve_assignment(bad)  # type: ignore[arg-type]

    def test_unmatched_observations_when_more_tracklets_than_candidates(self) -> None:
        v1 = _unit_vec(1)
        v2 = _unit_vec(2)
        self.memory.add_observation(v1, identity_id="alice", camera_id="cam_1", local_track_id=10, timestamp=100.0)

        associator = DataAssociator(memory=self.memory)
        tracklets = [
            TrackletObservation(local_track_id=1, camera_id="cam_1", timestamp=105.0, embedding=v1),
            TrackletObservation(local_track_id=2, camera_id="cam_1", timestamp=105.0, embedding=v2),
        ]
        result = associator.associate(tracklets)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].local_track_id, 1)
        self.assertEqual(result.matches[0].identity_id, "alice")
        self.assertEqual(result.matches[0].decision, ConfidenceDecision.ACCEPT)
        self.assertEqual([t.local_track_id for t in result.unmatched_tracklets], [2])

    def test_similar_looking_people_disambiguated_by_time_and_one_to_one(self) -> None:
        base = _unit_vec(10)
        twin_a = base + 0.05 * _unit_vec(11)
        twin_a = twin_a / np.linalg.norm(twin_a)
        twin_b = base + 0.05 * _unit_vec(12)
        twin_b = twin_b / np.linalg.norm(twin_b)

        self.memory.add_observation(twin_a, identity_id="person_a", camera_id="cam_1", local_track_id=1, timestamp=100.0)
        self.memory.add_observation(twin_b, identity_id="person_b", camera_id="cam_2", local_track_id=2, timestamp=195.0)

        rules = [
            CameraTransitionRule("cam_1", "cam_1", min_travel_sec=0.0, max_travel_sec=300.0),
            CameraTransitionRule("cam_2", "cam_2", min_travel_sec=0.0, max_travel_sec=300.0),
            CameraTransitionRule("cam_1", "cam_2", min_travel_sec=10.0, max_travel_sec=120.0),
            CameraTransitionRule("cam_2", "cam_1", min_travel_sec=10.0, max_travel_sec=120.0),
        ]
        associator = DataAssociator(memory=self.memory, transition_rules=rules)

        # Query at t=200 on cam_2 strongly matches person_b (dt=5s, same cam) vs person_a (dt=100s).
        # Query at t=200 on cam_1 cannot come from cam_2 in 5s (<10s min travel) -> must match person_a.
        tracklets = [
            TrackletObservation(local_track_id=50, camera_id="cam_1", timestamp=200.0, embedding=twin_a),
            TrackletObservation(local_track_id=51, camera_id="cam_2", timestamp=200.0, embedding=twin_b),
        ]
        result = associator.associate(tracklets)
        assigned = {m.local_track_id: m.identity_id for m in result.matches}
        self.assertEqual(assigned, {50: "person_a", 51: "person_b"})
        # Never merges global identities in IdentityMemory.
        self.assertEqual(len(self.memory.list_identities()), 2)

    def test_impossible_camera_transitions_gated(self) -> None:
        v = _unit_vec(20)
        self.memory.add_observation(v, identity_id="runner", camera_id="cam_a", timestamp=100.0)

        rules = [
            CameraTransitionRule("cam_a", "cam_b", min_travel_sec=30.0, max_travel_sec=120.0, allowed=True),
            CameraTransitionRule("cam_a", "cam_c", allowed=False),
        ]
        associator = DataAssociator(memory=self.memory, transition_rules=rules)

        # Too fast: dt=5s < min_travel_sec=30s
        too_fast = TrackletObservation(local_track_id=1, camera_id="cam_b", timestamp=105.0, embedding=v)
        res_fast = associator.associate([too_fast])
        self.assertEqual(res_fast.matches, [])
        self.assertEqual(len(res_fast.unmatched_tracklets), 1)

        # Forbidden transition: cam_a -> cam_c
        forbidden = TrackletObservation(local_track_id=2, camera_id="cam_c", timestamp=150.0, embedding=v)
        res_forb = associator.associate([forbidden])
        self.assertEqual(res_forb.matches, [])

    def test_temporal_constraints_negative_and_expired(self) -> None:
        v = _unit_vec(30)
        self.memory.add_observation(v, identity_id="traveler", camera_id="cam_1", timestamp=500.0)

        config = AssociationConfig(max_time_gap_sec=60.0)
        associator = DataAssociator(memory=self.memory, config=config)

        # Negative time gap (query earlier than stored observation)
        past_tracklet = TrackletObservation(local_track_id=1, camera_id="cam_1", timestamp=490.0, embedding=v)
        self.assertEqual(associator.associate([past_tracklet]).matches, [])

        # Exceeds max_time_gap_sec
        late_tracklet = TrackletObservation(local_track_id=2, camera_id="cam_1", timestamp=600.0, embedding=v)
        self.assertEqual(associator.associate([late_tracklet]).matches, [])

    def test_missing_and_uncalibrated_motion_not_invented(self) -> None:
        v = _unit_vec(40)
        self.memory.add_observation(v, identity_id="walker", camera_id="cam_1", timestamp=100.0)
        candidates = DataAssociator(memory=self.memory).retrieve_candidates(
            [TrackletObservation(local_track_id=1, camera_id="cam_1", timestamp=110.0, embedding=v)]
        )
        # Attach uncalibrated pixel motion on different cameras — must NOT be compared.
        cand_with_pixel = IdentityCandidate(
            identity_id=candidates[0].identity_id,
            observations=candidates[0].observations,
            best_similarity=candidates[0].best_similarity,
            motion_state=KalmanMotionEstimate(position=(10.0, 20.0), coordinate_frame="pixel", is_calibrated=False),
        )
        tracklet_other_cam = TrackletObservation(
            local_track_id=5,
            camera_id="cam_2",
            timestamp=110.0,
            embedding=v,
            motion_state=KalmanMotionEstimate(position=(900.0, 800.0), coordinate_frame="pixel", is_calibrated=False),
        )
        builder = CostMatrixBuilder(
            transition_rules=[CameraTransitionRule("cam_1", "cam_2", min_travel_sec=5.0, max_travel_sec=60.0)]
        )
        ev = builder.evaluate_pair(tracklet_other_cam, cand_with_pixel)
        self.assertFalse(ev.gated)
        self.assertIsNone(ev.motion_score)

    def test_calibrated_motion_gating(self) -> None:
        v = _unit_vec(45)
        self.memory.add_observation(v, identity_id="sprinter", camera_id="cam_1", timestamp=100.0)
        cands = DataAssociator(memory=self.memory).retrieve_candidates(
            [TrackletObservation(local_track_id=1, camera_id="cam_1", timestamp=102.0, embedding=v)]
        )
        cand_cal = IdentityCandidate(
            identity_id=cands[0].identity_id,
            observations=cands[0].observations,
            best_similarity=cands[0].best_similarity,
            motion_state=KalmanMotionEstimate(
                position=(0.0, 0.0), velocity=(1.0, 0.0), coordinate_frame="world", is_calibrated=True
            ),
        )
        far_tracklet = TrackletObservation(
            local_track_id=9,
            camera_id="cam_2",
            timestamp=102.0,
            embedding=v,
            motion_state=KalmanMotionEstimate(
                position=(500.0, 500.0), coordinate_frame="world", is_calibrated=True
            ),
        )
        builder = CostMatrixBuilder(
            config=AssociationConfig(max_motion_distance=50.0),
            transition_rules=[CameraTransitionRule("cam_1", "cam_2", min_travel_sec=0.0, max_travel_sec=60.0)],
        )
        ev = builder.evaluate_pair(far_tracklet, cand_cal)
        self.assertTrue(ev.gated)
        self.assertEqual(ev.gate_reason, "exceeds_max_motion_distance")
        self.assertEqual(ev.decision, ConfidenceDecision.REJECT)

    def test_uncertain_match_on_unknown_camera_transition(self) -> None:
        v = _unit_vec(50)
        self.memory.add_observation(v, identity_id="mystery", camera_id="cam_alpha", timestamp=100.0)

        # No transition rule between cam_alpha and cam_beta -> unknown transition -> UNCERTAIN
        associator = DataAssociator(memory=self.memory, transition_rules=[])
        tracklet = TrackletObservation(local_track_id=3, camera_id="cam_beta", timestamp=115.0, embedding=v)
        result = associator.associate([tracklet])
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].decision, ConfidenceDecision.UNCERTAIN)
        self.assertTrue(result.matches[0].evaluation.is_camera_uncertain)
        self.assertIsNone(result.matches[0].evaluation.camera_score)

    def test_rejected_match_on_weak_evidence(self) -> None:
        v1 = _unit_vec(60)
        v2 = -v1  # opposite direction -> cosine sim = -1.0 -> clipped to 0.0
        self.memory.add_observation(v1, identity_id="target", camera_id="cam_1", timestamp=100.0)

        # Force candidate evaluation directly even with low similarity
        obs = self.memory.get_identity_observations("target")
        cand = IdentityCandidate(identity_id="target", observations=obs, best_similarity=0.0)
        tracklet = TrackletObservation(local_track_id=4, camera_id="cam_1", timestamp=105.0, embedding=v2)

        associator = DataAssociator(memory=self.memory)
        result = associator.associate([tracklet], candidates=[cand])
        self.assertEqual(result.matches, [])
        self.assertEqual([t.local_track_id for t in result.unmatched_tracklets], [4])

    def test_multiple_historical_observations_per_identity(self) -> None:
        v_old = _unit_vec(70)
        v_new = _unit_vec(71)
        self.memory.add_observation(v_old, identity_id="multi", camera_id="cam_1", local_track_id=1, timestamp=100.0)
        self.memory.add_observation(v_new, identity_id="multi", camera_id="cam_1", local_track_id=2, timestamp=120.0)

        associator = DataAssociator(memory=self.memory)
        tracklet = TrackletObservation(local_track_id=8, camera_id="cam_1", timestamp=125.0, embedding=v_old)
        candidates = associator.retrieve_candidates([tracklet])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].observations), 2)

        result = associator.associate([tracklet])
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].identity_id, "multi")
        self.assertEqual(result.matches[0].decision, ConfidenceDecision.ACCEPT)


if __name__ == "__main__":
    unittest.main()
