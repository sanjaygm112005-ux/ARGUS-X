"""Unit tests for Milestone 11 & 13 evaluation metrics and bottleneck analysis (synthetic data)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.benchmark import BenchmarkResult, CameraBenchmarkStats, detect_hardware_info
from evaluation.metrics import (
    GroundTruthAnnotation,
    PredictedObservation,
    evaluate_predictions,
    load_ground_truth,
    save_ground_truth_template,
)
from evaluation.real_video_validation import analyze_bottlenecks
from evaluation.report_generator import generate_markdown_report


class TestEvaluationMetrics(unittest.TestCase):
    """Synthetic unit tests for frame-level and event-level metrics and reporting."""

    def test_perfect_recovery_frame_and_event_metrics(self) -> None:
        gt = [
            GroundTruthAnnotation("cam_1", 1.0, "p1", (10, 10, 50, 100), "camera_transition"),
            GroundTruthAnnotation("cam_1", 1.5, "p1", (12, 10, 52, 100), "camera_transition"),
            GroundTruthAnnotation("cam_2", 5.0, "p1", (20, 10, 60, 100), "camera_transition"),
            GroundTruthAnnotation("cam_2", 5.5, "p1", (22, 10, 62, 100), "camera_transition"),
            GroundTruthAnnotation("cam_1", 2.0, "p2", (80, 10, 120, 100), "short_occlusion"),
        ]
        preds = [
            PredictedObservation("cam_1", 1.0, "gid_a", 1, (10, 10, 50, 100), "NEW_IDENTITY"),
            PredictedObservation("cam_1", 1.5, "gid_a", 1, (12, 10, 52, 100), "ACTIVE_TRACK"),
            PredictedObservation("cam_2", 5.0, "gid_a", 4, (20, 10, 60, 100), "RECOVERED_ACCEPT"),
            PredictedObservation("cam_2", 5.5, "gid_a", 4, (22, 10, 62, 100), "ACTIVE_TRACK"),
            PredictedObservation("cam_1", 2.0, "gid_b", 2, (80, 10, 120, 100), "NEW_IDENTITY"),
        ]
        m = evaluate_predictions(gt, preds)
        self.assertEqual(m.total_ground_truth, 5)
        self.assertEqual(m.id_switches, 0)
        self.assertEqual(m.false_identity_matches, 0)
        self.assertEqual(m.missed_identity_recoveries, 0)
        self.assertAlmostEqual(m.identity_recovery_accuracy, 1.0)
        self.assertAlmostEqual(m.precision, 1.0)
        self.assertAlmostEqual(m.recall, 1.0)
        self.assertAlmostEqual(m.f1_score, 1.0)

        # Verify event-level separation: 1 discrete transition event (cam_1 -> cam_2),
        # even though cam_2 has 2 frame-level observations.
        assert m.event_metrics is not None
        self.assertEqual(m.event_metrics.total_recovery_events, 1)
        self.assertEqual(m.event_metrics.correct_recoveries, 1)
        self.assertAlmostEqual(m.event_metrics.recovery_event_accuracy, 1.0)
        self.assertIn("camera_transition", m.scenario_breakdown)

    def test_id_switches_false_matches_and_misses_recall_uninflated(self) -> None:
        gt = [
            GroundTruthAnnotation("cam_1", 1.0, "p1", (0, 0, 50, 100), "long_occlusion"),
            GroundTruthAnnotation("cam_1", 1.5, "p1", (0, 0, 50, 100), "long_occlusion"),
            GroundTruthAnnotation("cam_1", 5.0, "p1", (0, 0, 50, 100), "long_occlusion"),
            GroundTruthAnnotation("cam_2", 10.0, "p1", (0, 0, 50, 100), "long_occlusion"),
        ]
        preds = [
            PredictedObservation("cam_1", 1.0, "gid_1", 1, (0, 0, 50, 100), "NEW_IDENTITY"),
            PredictedObservation("cam_1", 1.5, "gid_1", 1, (0, 0, 50, 100), "ACTIVE_TRACK"),
            PredictedObservation("cam_1", 5.0, "gid_2", 2, (0, 0, 50, 100), "REJECT_NEW_IDENTITY"),
            PredictedObservation("cam_2", 10.0, None, 3, (0, 0, 50, 100), "UNCERTAIN"),
        ]
        m = evaluate_predictions(gt, preds)
        self.assertEqual(m.id_switches, 1)
        self.assertEqual(m.missed_identity_recoveries, 1)
        self.assertEqual(m.false_identity_matches, 1)
        self.assertEqual(m.true_positives, 2)
        # Recall = TP / total_ground_truth = 2/4 = 0.5
        self.assertAlmostEqual(m.recall, 0.5)
        self.assertAlmostEqual(m.precision, 2.0 / 3.0)

        assert m.event_metrics is not None
        self.assertEqual(m.event_metrics.total_recovery_events, 2)
        self.assertEqual(m.event_metrics.false_recoveries, 1)
        self.assertEqual(m.event_metrics.missed_recoveries, 1)
        self.assertAlmostEqual(m.event_metrics.recovery_event_accuracy, 0.0)

    def test_empty_ground_truth(self) -> None:
        m = evaluate_predictions([], [])
        self.assertEqual(m.total_ground_truth, 0)
        self.assertEqual(m.f1_score, 0.0)
        assert m.event_metrics is not None
        self.assertEqual(m.event_metrics.total_recovery_events, 0)

    def test_load_and_save_ground_truth_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "gt.json"
            save_ground_truth_template(p)
            self.assertEqual(load_ground_truth(p), [])

            p.write_text(
                json.dumps(
                    {
                        "annotations": [
                            {
                                "camera_id": "cam_1",
                                "timestamp": 1.5,
                                "gt_identity_id": "alice",
                                "bbox": [1, 2, 30, 60],
                                "scenario_tag": "similar_clothing",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_ground_truth(p)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].gt_identity_id, "alice")
            self.assertEqual(loaded[0].scenario_tag, "similar_clothing")

    def test_hardware_detection_bottleneck_and_report_generation(self) -> None:
        hw = detect_hardware_info()
        self.assertIn(hw.active_device, ("mps", "cpu"))
        self.assertGreater(hw.total_ram_gb, 0.0)

        res = BenchmarkResult(
            hardware=hw,
            camera_stats=[
                CameraBenchmarkStats("cam_1", "sample1.mp4", 10, 20.0, 18.0, 30.0, 2.0, 80.0)
            ],
            overall_detection_fps=20.0,
            overall_tracking_fps=18.0,
            avg_reid_extraction_ms=30.0,
            avg_association_ms=2.0,
            avg_e2e_latency_ms=80.0,
            cpu_percent=35.0,
            peak_memory_mb=420.0,
            is_real_video=False,
            video_sources=[],
            metrics=None,
        )
        bottleneck = analyze_bottlenecks(res)
        self.assertEqual(bottleneck.primary_bottleneck, "YOLO11n + BoT-SORT Tracking")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.md"
            md = generate_markdown_report(res, output_path=out, bottleneck=bottleneck)
            self.assertTrue(out.exists())
            self.assertIn("ARGUS-X Evaluation & Validation Report", md)
            self.assertIn("Synthetic / Unit Test Run", md)
            self.assertIn(hw.active_device, md)


if __name__ == "__main__":
    unittest.main()
