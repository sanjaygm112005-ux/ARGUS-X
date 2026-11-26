"""Unit tests for ground-truth annotation schema and geometry validation."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.annotation_validator import (
    VideoMetadata,
    validate_annotations_data,
    validate_annotations_file,
)


class TestAnnotationValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = {
            "camera1": VideoMetadata(
                camera_id="camera1",
                path=Path("datasets/camera1.mp4"),
                width=478,
                height=850,
                fps=30.0,
                frame_count=475,
                duration_sec=15.833,
            ),
            "camera2": VideoMetadata(
                camera_id="camera2",
                path=Path("datasets/camera2.mp4"),
                width=478,
                height=850,
                fps=30.01,
                frame_count=492,
                duration_sec=16.395,
            ),
        }

    def test_valid_annotations_pass(self) -> None:
        payload = {
            "annotations": [
                {
                    "camera_id": "camera1",
                    "timestamp": 2.5,
                    "gt_identity_id": "PERSON_A",
                    "bbox": [100.0, 120.0, 220.0, 500.0],
                    "scenario_tag": "camera_transition",
                },
                {
                    "camera_id": "camera2",
                    "timestamp": 8.0,
                    "gt_identity_id": "PERSON_A",
                    "bbox": [150.0, 140.0, 260.0, 520.0],
                    "scenario_tag": "camera_transition",
                },
            ]
        }
        errors = validate_annotations_data(payload, video_metadata=self.meta, require_non_empty=True)
        self.assertEqual(errors, [])

    def test_rejects_placeholder_coordinates_and_labels(self) -> None:
        payload = {
            "annotations": [
                {
                    "camera_id": "camera1",
                    "timestamp": 1.0,
                    "gt_identity_id": "TODO",
                    "bbox": [0, 0, 0, 0],
                }
            ]
        }
        errors = validate_annotations_data(payload, video_metadata=self.meta, require_non_empty=True)
        self.assertGreaterEqual(len(errors), 2)

    def test_rejects_out_of_bounds_bbox_and_duration(self) -> None:
        payload = {
            "annotations": [
                {
                    "camera_id": "camera1",
                    "timestamp": 25.0,
                    "gt_identity_id": "PERSON_A",
                    "bbox": [100.0, 100.0, 600.0, 900.0],
                }
            ]
        }
        errors = validate_annotations_data(payload, video_metadata=self.meta, require_non_empty=True)
        self.assertTrue(any("duration" in e for e in errors))
        self.assertTrue(any("resolution" in e for e in errors))

    def test_rejects_invalid_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{not valid json", encoding="utf-8")
            ok, errors = validate_annotations_file(p)
            self.assertFalse(ok)
            self.assertTrue(any("Invalid JSON" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
