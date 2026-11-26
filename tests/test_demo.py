"""Tests for Milestone 12: End-to-End Multi-Camera Demonstration Workflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from demo.run_demo import DemoSummary, MultiCameraDemoRunner, create_sample_demo_videos


class TestMultiCameraDemo(unittest.TestCase):
    """End-to-end demonstration workflow test using synthetic sample video clips."""

    def test_end_to_end_demo_pipeline_with_synthetic_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cam_videos, gt_path = create_sample_demo_videos(Path(tmpdir))
            runner = MultiCameraDemoRunner()
            try:
                summary = runner.run(
                    cam_videos,
                    camera_start_offsets_sec={"cam_1": 0.0, "cam_2": 7.0},
                    ground_truth_path=gt_path,
                    is_synthetic_footage=True,
                )
                self.assertIsInstance(summary, DemoSummary)
                self.assertEqual(summary.videos_processed, 2)
                self.assertEqual(summary.total_frames_processed, 10)
                self.assertTrue(summary.verified_against_ground_truth)
                self.assertIsNotNone(summary.metrics)
                self.assertGreater(summary.overall_fps, 0.0)
                self.assertGreater(summary.audit_log_count, 0)
                # Ensure dashboard/API outputs never expose raw embeddings
                for cam in runner.service.list_cameras():
                    for det in cam["active_detections"]:
                        self.assertNotIn("embedding", det)
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
