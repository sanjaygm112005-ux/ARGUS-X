"""Unit tests for the modular YOLO11n Detector."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection import Detection, Detector


class TestDetector(unittest.TestCase):
    """Verify YOLO11n detection on images, video frames, thresholds, and error cases."""

    detector: Detector
    sample_image: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = Detector(model_path="yolo11n.pt", conf_threshold=0.4)
        cls.sample_image = PROJECT_ROOT / "bus.jpg"

    def test_device_and_single_model_instance(self) -> None:
        self.assertIn(self.detector.device, ("mps", "cpu"))
        initial_model = self.detector.model
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        self.detector.detect(frame)
        self.assertIs(self.detector.model, initial_model)

    def test_detect_image_file(self) -> None:
        if not self.sample_image.is_file():
            self.skipTest("Sample image bus.jpg is not present in workspace root.")

        detections = self.detector.detect(self.sample_image)
        self.assertGreater(len(detections), 0)
        for det in detections:
            self.assertIsInstance(det, Detection)
            self.assertEqual(len(det.bbox), 4)
            x1, y1, x2, y2 = det.bbox
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertIsInstance(det.class_id, int)
            self.assertIsInstance(det.class_name, str)
            self.assertGreaterEqual(det.confidence, 0.4)
            self.assertLessEqual(det.confidence, 1.0)

        names = {det.class_name for det in detections}
        self.assertIn("bus", names)
        self.assertIn("person", names)

    def test_detect_video_frame_ndarray(self) -> None:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blank_detections = self.detector.detect(blank_frame)
        self.assertIsInstance(blank_detections, list)
        self.assertEqual(len(blank_detections), 0)

        if self.sample_image.is_file():
            frame = cv2.imread(str(self.sample_image))
            self.assertIsNotNone(frame)
            frame_detections = self.detector.detect(frame)
            self.assertGreater(len(frame_detections), 0)

    def test_configurable_confidence_threshold(self) -> None:
        if not self.sample_image.is_file():
            self.skipTest("Sample image bus.jpg is not present in workspace root.")

        low_conf_dets = self.detector.detect(self.sample_image, conf_threshold=0.2)
        high_conf_dets = self.detector.detect(self.sample_image, conf_threshold=0.8)
        self.assertGreaterEqual(len(low_conf_dets), len(high_conf_dets))
        for det in high_conf_dets:
            self.assertGreaterEqual(det.confidence, 0.8)

    def test_missing_model_and_invalid_inputs(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Detector(model_path="nonexistent_yolo11n.pt")

        with self.assertRaises(ValueError):
            Detector(model_path="yolo11n.pt", conf_threshold=1.5)

        with self.assertRaises(FileNotFoundError):
            self.detector.detect("does_not_exist.jpg")

        with self.assertRaises(ValueError):
            self.detector.detect(None)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            self.detector.detect(np.array([], dtype=np.uint8))

        with self.assertRaises(ValueError):
            self.detector.detect(self.sample_image, conf_threshold=-0.1)

        with self.assertRaises(TypeError):
            self.detector.detect(12345)  # type: ignore[arg-type]

        with tempfile.NamedTemporaryFile(suffix=".jpg") as bad_file:
            bad_file.write(b"not a valid image")
            bad_file.flush()
            with self.assertRaises(ValueError):
                self.detector.detect(bad_file.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
