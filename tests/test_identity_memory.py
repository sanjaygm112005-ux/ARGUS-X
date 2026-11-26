"""Tests for Milestone 5: Long-Term Identity Memory (FAISS + SQLite)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from identity_memory import IdentityMemory, ObservationRecord, SearchMatch


def _unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestIdentityMemory(unittest.TestCase):
    """Unit tests for IdentityMemory and SQLite IdentityStorage."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.db_path = self.tmp_path / "memory.db"
        self.index_path = self.tmp_path / "memory.faiss"
        self.memory = IdentityMemory(
            embedding_dim=512,
            db_path=self.db_path,
            index_path=self.index_path,
        )

    def tearDown(self) -> None:
        self.memory.close()
        self._tmpdir.cleanup()

    def test_add_and_search_single_embedding(self) -> None:
        vec = _unit_vec(1)
        rec = self.memory.add_observation(
            embedding=vec,
            identity_id="person_a",
            camera_id="cam_1",
            local_track_id=7,
            timestamp=1700000000.0,
            embedding_id="emb_001",
        )
        self.assertIsInstance(rec, ObservationRecord)
        self.assertEqual(self.memory.size, 1)
        self.assertEqual(self.memory.vector_to_identity[rec.vector_id], "person_a")

        matches = self.memory.search(vec, top_k=1)
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertIsInstance(m, SearchMatch)
        self.assertEqual(m.identity_id, "person_a")
        self.assertEqual(m.camera_id, "cam_1")
        self.assertEqual(m.local_track_id, 7)
        self.assertEqual(m.embedding_id, "emb_001")
        self.assertAlmostEqual(m.similarity, 1.0, places=5)

    def test_top_k_retrieval_and_ordering(self) -> None:
        base = _unit_vec(10)
        near = base + 0.1 * _unit_vec(11)
        near = near / np.linalg.norm(near)
        far = -base

        self.memory.add_observation(far, identity_id="id_far", camera_id="cam_0")
        self.memory.add_observation(near, identity_id="id_near", camera_id="cam_1")
        self.memory.add_observation(base, identity_id="id_exact", camera_id="cam_2")

        matches = self.memory.search(base, top_k=2)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].identity_id, "id_exact")
        self.assertEqual(matches[1].identity_id, "id_near")
        self.assertGreaterEqual(matches[0].similarity, matches[1].similarity)

    def test_multiple_embeddings_per_identity_retained(self) -> None:
        v1 = _unit_vec(20)
        v2 = _unit_vec(21)
        v3 = _unit_vec(22)

        self.memory.add_observation(v1, identity_id="person_x", camera_id="cam_1", local_track_id=1, timestamp=100.0)
        self.memory.add_observation(v2, identity_id="person_x", camera_id="cam_1", local_track_id=1, timestamp=105.0)
        self.memory.add_observation(v3, identity_id="person_x", camera_id="cam_2", local_track_id=4, timestamp=200.0)

        obs = self.memory.get_identity_observations("person_x")
        self.assertEqual(len(obs), 3)
        self.assertEqual([o.timestamp for o in obs], [100.0, 105.0, 200.0])
        self.assertEqual(self.memory.size, 3)

    def test_never_auto_merges_identities_on_similarity(self) -> None:
        vec = _unit_vec(30)
        self.memory.add_observation(vec, identity_id="person_1", camera_id="cam_1", local_track_id=1)
        self.memory.add_observation(vec, identity_id="person_2", camera_id="cam_2", local_track_id=1)

        self.assertEqual(len(self.memory.list_identities()), 2)
        matches = self.memory.search(vec, top_k=2)
        returned_ids = {m.identity_id for m in matches}
        self.assertEqual(returned_ids, {"person_1", "person_2"})

    def test_normalization_of_unnormalized_vectors(self) -> None:
        raw = _unit_vec(40) * 25.0
        self.memory.add_observation(raw, identity_id="scaled_person", camera_id="cam_0")

        query = raw * 0.05
        matches = self.memory.search(query, top_k=1)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].similarity, 1.0, places=5)

    def test_duplicate_embedding_handling(self) -> None:
        vec = _unit_vec(50)
        r1 = self.memory.add_observation(vec, identity_id="dup_person", camera_id="cam_0")
        r2 = self.memory.add_observation(vec, identity_id="dup_person", camera_id="cam_0")

        self.assertEqual(r1.embedding_id, r2.embedding_id)
        self.assertEqual(self.memory.size, 1)

        with self.assertRaises(ValueError):
            self.memory.add_observation(
                _unit_vec(51),
                identity_id="dup_person",
                camera_id="cam_0",
                embedding_id=r1.embedding_id,
            )

    def test_invalid_and_empty_embeddings(self) -> None:
        with self.assertRaises(ValueError):
            self.memory.add_observation(np.zeros(512, dtype=np.float32), identity_id="p1")

        with self.assertRaises(ValueError):
            self.memory.add_observation(np.ones(256, dtype=np.float32), identity_id="p1")

        bad_nan = _unit_vec(60)
        bad_nan[0] = np.nan
        with self.assertRaises(ValueError):
            self.memory.add_observation(bad_nan, identity_id="p1")

        with self.assertRaises(ValueError):
            self.memory.add_observation(np.array([], dtype=np.float32), identity_id="p1")

        empty_batch = self.memory.add_observations_batch(
            np.empty((0, 512), dtype=np.float32),
            identity_ids=[],
        )
        self.assertEqual(empty_batch, [])
        self.assertEqual(self.memory.search(np.empty((0, 512), dtype=np.float32)), [])
        self.assertEqual(self.memory.search(_unit_vec(61), top_k=5), [])

    def test_save_load_and_restart_recovery(self) -> None:
        v1 = _unit_vec(70)
        v2 = _unit_vec(71)
        self.memory.add_observation(
            v1, identity_id="alice", camera_id="cam_north", local_track_id=12, timestamp=1001.5
        )
        self.memory.add_observation(
            v2, identity_id="bob", camera_id="cam_south", local_track_id=3, timestamp=1002.5
        )
        saved_path = self.memory.save_index()
        self.assertTrue(saved_path.exists())
        self.memory.close()

        restarted = IdentityMemory(
            embedding_dim=512,
            db_path=self.db_path,
            index_path=self.index_path,
        )
        try:
            self.assertEqual(restarted.size, 2)
            self.assertEqual(set(restarted.list_identities()), {"alice", "bob"})
            matches = restarted.search(v1, top_k=1)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].identity_id, "alice")
            self.assertEqual(matches[0].camera_id, "cam_north")
            self.assertEqual(matches[0].local_track_id, 12)
            self.assertAlmostEqual(matches[0].timestamp, 1001.5, places=4)
            self.assertAlmostEqual(matches[0].similarity, 1.0, places=5)
        finally:
            restarted.close()


if __name__ == "__main__":
    unittest.main()
