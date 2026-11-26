"""Unit tests for the modular YOLO11n + BoT-SORT ObjectTracker."""

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

from tracking import ObjectTracker, TrackedObject


def build_shifted_frames(image: np.ndarray, num_frames: int, step_px: int = 4) -> list[np.ndarray]:
    """Create consecutive translated frames to simulate smooth camera/object motion."""
    height, width = image.shape[:2]
    frames: list[np.ndarray] = []
    for idx in range(num_frames):
        tx = float(idx * step_px)
        matrix = np.float32([[1, 0, tx], [0, 1, 0]])
        shifted = cv2.warpAffine(image, matrix, (width, height))
        frames.append(shifted)
    return frames


class TestObjectTracker(unittest.TestCase):
    """Verify BoT-SORT local tracking, ID stability, stream input, and error handling."""

    tracker: ObjectTracker
    sample_image: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tracker = ObjectTracker(
            model_path="yolo11n.pt",
            tracker_config="botsort.yaml",
            conf_threshold=0.4,
        )
        cls.sample_image = PROJECT_ROOT / "bus.jpg"

    def setUp(self) -> None:
        self.tracker.reset()

    def test_device_and_botsort_config(self) -> None:
        self.assertIn(self.tracker.device, ("mps", "cpu"))
        self.assertTrue(self.tracker.tracker_config.endswith("botsort.yaml"))
        self.assertTrue(Path(self.tracker.tracker_config).is_file())

    def test_multiple_objects_and_stable_local_ids(self) -> None:
        if not self.sample_image.is_file():
            self.skipTest("Sample image bus.jpg is not present in workspace root.")

        base_img = cv2.imread(str(self.sample_image))
        self.assertIsNotNone(base_img)

        frames = build_shifted_frames(base_img, num_frames=6, step_px=3)
        per_frame_ids: list[set[int]] = []

        for frame in frames:
            tracks = self.tracker.track_frame(frame)
            self.assertGreaterEqual(len(tracks), 2)
            for item in tracks:
                self.assertIsInstance(item, TrackedObject)
                self.assertIsInstance(item.track_id, int)
                self.assertGreater(item.track_id, 0)
                x1, y1, x2, y2 = item.bbox
                self.assertLess(x1, x2)
                self.assertLess(y1, y2)
                self.assertIsInstance(item.class_name, str)
                self.assertGreaterEqual(item.confidence, 0.4)
                self.assertLessEqual(item.confidence, 1.0)
            per_frame_ids.append({item.track_id for item in tracks})

        # Local track IDs should remain consistent across consecutive frames.
        persistent_ids = set.intersection(*per_frame_ids)
        self.assertGreaterEqual(len(persistent_ids), 2)

    def test_objects_entering_and_leaving_frame(self) -> None:
        if not self.sample_image.is_file():
            self.skipTest("Sample image bus.jpg is not present in workspace root.")

        base_img = cv2.imread(str(self.sample_image))
        self.assertIsNotNone(base_img)
        blank = np.zeros_like(base_img)

        # Start empty, objects enter across consecutive frames, then leave the frame.
        self.assertEqual(self.tracker.track_frame(blank), [])

        # First frame after an empty frame initializes tentative tracks; second confirms them.
        self.tracker.track_frame(base_img)
        entered_tracks = self.tracker.track_frame(base_img)
        self.assertGreater(len(entered_tracks), 0)

        exited_tracks = self.tracker.track_frame(blank)
        self.assertEqual(exited_tracks, [])

    def test_video_file_stream_tracking(self) -> None:
        if not self.sample_image.is_file():
            self.skipTest("Sample image bus.jpg is not present in workspace root.")

        base_img = cv2.imread(str(self.sample_image))
        self.assertIsNotNone(base_img)
        height, width = base_img.shape[:2]
        frames = build_shifted_frames(base_img, num_frames=5, step_px=3)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_video.name, fourcc, 10.0, (width, height))
            self.assertTrue(writer.isOpened())
            try:
                for frame in frames:
                    writer.write(frame)
            finally:
                writer.release()

            seen_frames = 0
            all_ids: list[set[int]] = []
            for frame_idx, _, tracks in self.tracker.track_stream(tmp_video.name):
                self.assertEqual(frame_idx, seen_frames)
                self.assertGreater(len(tracks), 0)
                all_ids.append({t.track_id for t in tracks})
                seen_frames += 1

            self.assertEqual(seen_frames, len(frames))
            self.assertGreater(len(set.intersection(*all_ids)), 0)

    def test_empty_frames_and_invalid_inputs(self) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(self.tracker.track_frame(blank), [])

        with self.assertRaises(FileNotFoundError):
            ObjectTracker(model_path="missing_model.pt")

        with self.assertRaises(FileNotFoundError):
            ObjectTracker(model_path="yolo11n.pt", tracker_config="missing_tracker.yaml")

        with self.assertRaises(ValueError):
            self.tracker.track_frame(None)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            self.tracker.track_frame(np.array([], dtype=np.uint8))

        with self.assertRaises(TypeError):
            self.tracker.track_frame("bus.jpg")  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            self.tracker.track_frame(blank, conf_threshold=1.5)

        with self.assertRaises(FileNotFoundError):
            list(self.tracker.track_stream("nonexistent_video.mp4"))

        with self.assertRaises(ValueError):
            list(self.tracker.track_stream(-1))

        with self.assertRaises(TypeError):
            list(self.tracker.track_stream(True))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
