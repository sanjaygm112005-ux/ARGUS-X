"""Real-world video validation, ground-truth metrics, and benchmarking for ARGUS-X."""

from evaluation.benchmark import (
    BenchmarkResult,
    CameraBenchmarkStats,
    HardwareInfo,
    MultiCameraBenchmarker,
    detect_hardware_info,
)
from evaluation.metrics import (
    EvaluationMetrics,
    GroundTruthAnnotation,
    PredictedObservation,
    evaluate_predictions,
    load_ground_truth,
    save_ground_truth_template,
)
from evaluation.report_generator import generate_markdown_report

__all__ = [
    "BenchmarkResult",
    "CameraBenchmarkStats",
    "EvaluationMetrics",
    "GroundTruthAnnotation",
    "HardwareInfo",
    "MultiCameraBenchmarker",
    "PredictedObservation",
    "detect_hardware_info",
    "evaluate_predictions",
    "generate_markdown_report",
    "load_ground_truth",
    "save_ground_truth_template",
]
