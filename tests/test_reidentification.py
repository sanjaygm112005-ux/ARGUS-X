"""Unit tests for PersonFeatureExtractor (OSNet appearance embeddings)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reidentification import PersonFeatureExtractor
from reidentification.feature_extractor import _MIN_CROP_HEIGHT, _MIN_CROP_WIDTH


class TestPersonFeatureExtractor(unittest.TestCase):
    """Verify OSNet embedding extraction, normalization, batch, device, and error handling."""

    extractor: PersonFeatureExtractor
    sample_image: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = PersonFeatureExtractor(
            model_name="osnet_x0_25",
            device="auto",
        )
        cls.sample_image = PROJECT_ROOT / "bus.jpg"

    def _make_dummy_crop(self, h: int = 256, w: int = 128) -> np.ndarray:
        """Create a random BGR crop simulating a person image."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    def test_device_selection(self) -> None:
        self.assertIn(self.extractor.device_str, ("mps", "cpu"))
        self.assertIsNotNone(self.extractor.device)
        self.assertEqual(str(self.extractor.device), self.extractor.device_str)

    def test_embedding_dimensions(self) -> None:
        crop = self._make_dummy_crop()
        embedding = self.extractor.extract(crop)
        self.assertEqual(embedding.ndim, 1)
        self.assertEqual(embedding.shape[0], self.extractor.embedding_dim)
        self.assertEqual(embedding.shape[0], 512)
        self.assertEqual(embedding.dtype, np.float32)

    def test_embedding_l2_normalization(self) -> None:
        crop = self._make_dummy_crop()
        embedding = self.extractor.extract(crop)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_embedding_repeatability(self) -> None:
        crop = self._make_dummy_crop()
        emb1 = self.extractor.extract(crop)
        emb2 = self.extractor.extract(crop)
        np.testing.assert_allclose(emb1, emb2, atol=1e-6)

    def test_different_crops_produce_different_embeddings(self) -> None:
        crop_a = self._make_dummy_crop(256, 128)
        crop_b = np.zeros((256, 128, 3), dtype=np.uint8)
        emb_a = self.extractor.extract(crop_a)
        emb_b = self.extractor.extract(crop_b)
        cosine_sim = float(np.dot(emb_a, emb_b))
        self.assertLess(cosine_sim, 0.99)

    def test_batch_extraction(self) -> None:
        crops = [self._make_dummy_crop(h=200, w=80) for _ in range(4)]
        embeddings = self.extractor.extract_batch(crops)
        self.assertEqual(embeddings.shape, (4, 512))
        for row_idx in range(4):
            norm = np.linalg.norm(embeddings[row_idx])
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_empty_batch_returns_empty_array(self) -> None:
        embeddings = self.extractor.extract_batch([])
        self.assertEqual(embeddings.shape, (0, 512))

    def test_extract_from_frame_with_bboxes(self) -> None:
        if not self.sample_image.is_file():
            self.skipTest("Sample image bus.jpg is not present in workspace root.")

        frame = cv2.imread(str(self.sample_image))
        self.assertIsNotNone(frame)
        bboxes = [(10, 50, 110, 350), (200, 80, 310, 400)]
        embeddings = self.extractor.extract_from_frame(frame, bboxes)
        self.assertEqual(embeddings.ndim, 2)
        self.assertGreaterEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 512)

    def test_extract_from_frame_skips_tiny_crops(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tiny_bbox = (0, 0, 5, 5)
        embeddings = self.extractor.extract_from_frame(frame, [tiny_bbox])
        self.assertEqual(embeddings.shape, (0, 512))

    def test_crop_person_clamps_to_frame_bounds(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        crop = PersonFeatureExtractor.crop_person(frame, (-10, -10, 100, 100))
        self.assertEqual(crop.shape, (100, 100, 3))

    def test_invalid_crop_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.extractor.extract(None)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            self.extractor.extract(np.array([], dtype=np.uint8))

        with self.assertRaises(ValueError):
            tiny = np.zeros((_MIN_CROP_HEIGHT - 1, _MIN_CROP_WIDTH, 3), dtype=np.uint8)
            self.extractor.extract(tiny)

        with self.assertRaises(ValueError):
            tiny = np.zeros((_MIN_CROP_HEIGHT, _MIN_CROP_WIDTH - 1, 3), dtype=np.uint8)
            self.extractor.extract(tiny)

        with self.assertRaises(ValueError):
            self.extractor.extract_batch([None])  # type: ignore[list-item]

    def test_unknown_model_raises(self) -> None:
        with self.assertRaises(ValueError):
            PersonFeatureExtractor(model_name="osnet_nonexistent")

    def test_missing_weights_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            PersonFeatureExtractor(
                model_name="osnet_x0_25",
                weights_path="/nonexistent/path/model.pth",
            )

    def test_varying_crop_sizes(self) -> None:
        """Different crop sizes should all produce valid 512-dim normalized embeddings."""
        sizes = [(64, 32), (128, 64), (256, 128), (512, 256)]
        for h, w in sizes:
            crop = self._make_dummy_crop(h, w)
            emb = self.extractor.extract(crop)
            self.assertEqual(emb.shape, (512,))
            self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
