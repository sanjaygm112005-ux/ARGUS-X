"""Reproducible real-world multi-camera CCTV validation workflow for ARGUS-X.

Requires at least two authorized local video files and (for verified accuracy)
a ground-truth JSON file. Explicitly separates real-video evaluation from
synthetic unit tests and identifies performance bottlenecks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from association import CameraTransitionRule
from evaluation.benchmark import BenchmarkResult, MultiCameraBenchmarker
from evaluation.report_generator import generate_markdown_report


@dataclass(frozen=True)
class BottleneckAnalysis:
    """Breakdown of per-stage latency share to identify pipeline bottlenecks."""

    primary_bottleneck: str
    reid_share_pct: float
    association_share_pct: float
    tracking_and_other_share_pct: float


def analyze_bottlenecks(result: BenchmarkResult) -> BottleneckAnalysis:
    """Identify the dominant latency stage from a BenchmarkResult."""
    e2e = max(1e-6, result.avg_e2e_latency_ms)
    reid_pct = (result.avg_reid_extraction_ms / e2e) * 100.0
    assoc_pct = (result.avg_association_ms / e2e) * 100.0
    other_pct = max(0.0, 100.0 - reid_pct - assoc_pct)

    shares = {
        "OSNet Re-ID Extraction": reid_pct,
        "Data Association & Memory": assoc_pct,
        "YOLO11n + BoT-SORT Tracking": other_pct,
    }
    primary = max(shares, key=lambda k: shares[k])
    return BottleneckAnalysis(
        primary_bottleneck=primary,
        reid_share_pct=round(reid_pct, 2),
        association_share_pct=round(assoc_pct, 2),
        tracking_and_other_share_pct=round(other_pct, 2),
    )


def run_real_video_validation(
    camera_videos: Sequence[tuple[str, Union[str, Path]]],
    ground_truth_path: Optional[Union[str, Path]] = None,
    transition_rules: Optional[Sequence[CameraTransitionRule]] = None,
    report_path: Union[str, Path] = "evaluation/results/real_video_validation.md",
    results_json_path: Union[str, Path] = "evaluation/results/real_video_validation.json",
    max_frames_per_camera: Optional[int] = None,
) -> tuple[BenchmarkResult, BottleneckAnalysis]:
    """Run real-video validation and generate a factual Markdown + JSON report."""
    benchmarker = MultiCameraBenchmarker(
        device="auto",
        transition_rules=transition_rules,
        max_frames_per_camera=max_frames_per_camera,
    )
    result = benchmarker.run_on_videos(
        camera_videos=camera_videos,
        ground_truth_path=ground_truth_path,
        results_json_path=results_json_path,
    )
    bottleneck = analyze_bottlenecks(result)
    generate_markdown_report(result, output_path=report_path, bottleneck=bottleneck)
    return result, bottleneck


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARGUS-X Real-World Multi-Camera Video Validation"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Authorized local camera-video pairs in cam_id:path format (minimum 2).",
    )
    parser.add_argument("--ground-truth", default=None, help="Ground-truth JSON path.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--report",
        default="evaluation/results/real_video_validation.md",
    )
    parser.add_argument(
        "--output-json",
        default="evaluation/results/real_video_validation.json",
    )
    args = parser.parse_args()

    cam_videos: list[tuple[str, str]] = []
    for item in args.videos:
        if ":" not in item:
            raise SystemExit(f"Invalid --videos entry '{item}', expected cam_id:path")
        cid, vpath = item.split(":", 1)
        cam_videos.append((cid, vpath))

    result, bottleneck = run_real_video_validation(
        camera_videos=cam_videos,
        ground_truth_path=args.ground_truth,
        report_path=args.report,
        results_json_path=args.output_json,
        max_frames_per_camera=args.max_frames,
    )
    print(
        json.dumps(
            {
                "benchmark": result.to_dict(),
                "bottleneck": bottleneck.__dict__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
