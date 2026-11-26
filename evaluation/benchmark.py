"""Reproducible multi-camera video benchmarking for ARGUS-X.

Measures detection FPS, BoT-SORT tracking FPS, OSNet Re-ID extraction time,
Hungarian/FAISS association time, end-to-end latency, and CPU/memory usage
on local video files without fabricating performance numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import psutil
import torch

from association import AssociationConfig, CameraTransitionRule
from config import resolve_device
from detection.detector import Detector
from evaluation.metrics import (
    EvaluationMetrics,
    PredictedObservation,
    evaluate_predictions,
    load_ground_truth,
)
from identity_memory import MultiCameraIdentityManager, TrackletManager
from reidentification.feature_extractor import PersonFeatureExtractor
from tracking.tracker import ObjectTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareInfo:
    """Detected host hardware and PyTorch compute device."""

    os_name: str
    machine_arch: str
    python_version: str
    pytorch_version: str
    active_device: str
    mps_built: bool
    mps_available: bool
    cpu_count_logical: int
    total_ram_gb: float


@dataclass(frozen=True)
class CameraBenchmarkStats:
    """Measured per-camera processing performance."""

    camera_id: str
    source: str
    frames_processed: int
    detection_fps: float
    tracking_fps: float
    avg_reid_ms: float
    avg_association_ms: float
    avg_e2e_latency_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete measured benchmark and optional ground-truth evaluation result."""

    hardware: HardwareInfo
    camera_stats: list[CameraBenchmarkStats]
    overall_detection_fps: float
    overall_tracking_fps: float
    avg_reid_extraction_ms: float
    avg_association_ms: float
    avg_e2e_latency_ms: float
    cpu_percent: float
    peak_memory_mb: float
    is_real_video: bool
    video_sources: list[str] = field(default_factory=list)
    metrics: Optional[EvaluationMetrics] = None
    failure_cases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_hardware_info(preferred_device: str = "auto") -> HardwareInfo:
    """Detect actual hardware and PyTorch device without assuming MPS."""
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    mps_avail = mps_built and torch.backends.mps.is_available()
    active = resolve_device(preferred_device)
    total_ram_gb = psutil.virtual_memory().total / (1024.0 ** 3)
    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()}",
        machine_arch=platform.machine(),
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        active_device=active,
        mps_built=mps_built,
        mps_available=mps_avail,
        cpu_count_logical=os.cpu_count() or 1,
        total_ram_gb=round(total_ram_gb, 2),
    )


class MultiCameraBenchmarker:
    """Runs reproducible multi-camera benchmarks on locally supplied video files."""

    def __init__(
        self,
        device: str = "auto",
        transition_rules: Optional[Sequence[CameraTransitionRule]] = None,
        max_frames_per_camera: Optional[int] = None,
    ) -> None:
        self.hardware = detect_hardware_info(device)
        self.transition_rules = list(transition_rules or [])
        self.max_frames_per_camera = max_frames_per_camera

    def run_on_videos(
        self,
        camera_videos: Sequence[tuple[str, Union[str, Path]]],
        ground_truth_path: Optional[Union[str, Path]] = None,
        results_json_path: Optional[Union[str, Path]] = None,
    ) -> BenchmarkResult:
        """Benchmark detection, tracking, Re-ID, and association on local videos."""
        if len(camera_videos) < 2:
            raise ValueError(
                "Multi-camera benchmark requires at least 2 local video sources."
            )

        resolved_sources: list[tuple[str, Path]] = []
        for cam_id, raw_path in camera_videos:
            p = Path(raw_path)
            if not p.exists():
                raise FileNotFoundError(f"Video file not found for {cam_id}: {p}")
            resolved_sources.append((cam_id, p))

        detector = Detector(device=self.hardware.active_device)
        extractor = PersonFeatureExtractor(device=self.hardware.active_device)
        mc_mgr = MultiCameraIdentityManager(
            transition_rules=self.transition_rules,
            config=AssociationConfig(max_time_gap_sec=3600.0),
        )

        proc = psutil.Process(os.getpid())
        proc.cpu_percent(interval=None)
        peak_rss_mb = proc.memory_info().rss / (1024.0 ** 2)

        predictions: list[PredictedObservation] = []
        cam_stats: list[CameraBenchmarkStats] = []
        total_det_sec = 0.0
        total_trk_sec = 0.0
        total_reid_sec = 0.0
        total_assoc_sec = 0.0
        total_e2e_sec = 0.0
        total_frames = 0
        failure_cases: list[str] = []

        try:
            for cam_id, video_path in resolved_sources:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video file: {video_path}")

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                tracker = ObjectTracker(device=self.hardware.active_device)
                tm = TrackletManager(
                    camera_id=cam_id,
                    feature_extractor=None,
                    inactive_timeout_sec=1.0,
                    extraction_interval_sec=0.5,
                )

                c_frames = 0
                c_det_sec = 0.0
                c_trk_sec = 0.0
                c_reid_sec = 0.0
                c_assoc_sec = 0.0
                c_e2e_sec = 0.0

                try:
                    while True:
                        if (
                            self.max_frames_per_camera is not None
                            and c_frames >= self.max_frames_per_camera
                        ):
                            break
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break

                        ts = c_frames / fps
                        t_e2e_start = time.perf_counter()

                        t0 = time.perf_counter()
                        _ = detector.detect(frame)
                        t1 = time.perf_counter()
                        c_det_sec += t1 - t0

                        t2 = time.perf_counter()
                        tracked = tracker.track_frame(frame)
                        t3 = time.perf_counter()
                        c_trk_sec += t3 - t2

                        t4 = time.perf_counter()
                        person_objs = [o for o in tracked if o.class_id == 0]
                        precomputed: dict[int, np.ndarray] = {}
                        if person_objs:
                            valid_crops = []
                            valid_tids = []
                            for o in person_objs:
                                try:
                                    valid_crops.append(extractor.crop_person(frame, o.bbox))
                                    valid_tids.append(o.track_id)
                                except ValueError:
                                    continue
                            if valid_crops:
                                feats = extractor.extract_batch(valid_crops)
                                for feat_row, tid in zip(feats, valid_tids):
                                    precomputed[tid] = feat_row
                        t5 = time.perf_counter()
                        c_reid_sec += t5 - t4

                        t6 = time.perf_counter()
                        upd = tm.update(
                            tracked, timestamp=ts, precomputed_embeddings=precomputed
                        )
                        outs = mc_mgr.process_tracklets(
                            upd.active_tracklets, current_timestamp=ts
                        )
                        t7 = time.perf_counter()
                        c_assoc_sec += t7 - t6

                        c_e2e_sec += time.perf_counter() - t_e2e_start
                        c_frames += 1

                        for out in outs:
                            predictions.append(
                                PredictedObservation(
                                    camera_id=out.camera_id,
                                    timestamp=ts,
                                    predicted_global_id=out.global_id,
                                    local_track_id=out.local_track_id,
                                    bbox=out.bounding_box,
                                    recovery_decision=out.recovery_decision,
                                )
                            )
                            if out.recovery_decision == "UNCERTAIN":
                                failure_cases.append(
                                    f"t={ts:.2f}s {cam_id} local={out.local_track_id}: UNCERTAIN"
                                )

                        rss_mb = proc.memory_info().rss / (1024.0 ** 2)
                        if rss_mb > peak_rss_mb:
                            peak_rss_mb = rss_mb
                finally:
                    cap.release()
                    mc_mgr.handle_camera_disconnect(cam_id, timestamp=c_frames / fps)

                if c_frames > 0:
                    cam_stats.append(
                        CameraBenchmarkStats(
                            camera_id=cam_id,
                            source=str(video_path),
                            frames_processed=c_frames,
                            detection_fps=c_frames / c_det_sec if c_det_sec > 0 else 0.0,
                            tracking_fps=c_frames / c_trk_sec if c_trk_sec > 0 else 0.0,
                            avg_reid_ms=(c_reid_sec / c_frames) * 1000.0,
                            avg_association_ms=(c_assoc_sec / c_frames) * 1000.0,
                            avg_e2e_latency_ms=(c_e2e_sec / c_frames) * 1000.0,
                        )
                    )
                total_frames += c_frames
                total_det_sec += c_det_sec
                total_trk_sec += c_trk_sec
                total_reid_sec += c_reid_sec
                total_assoc_sec += c_assoc_sec
                total_e2e_sec += c_e2e_sec
        finally:
            mc_mgr.close()

        cpu_pct = proc.cpu_percent(interval=None)
        metrics: Optional[EvaluationMetrics] = None
        if ground_truth_path is not None:
            gt = load_ground_truth(ground_truth_path)
            metrics = evaluate_predictions(gt, predictions)

        result = BenchmarkResult(
            hardware=self.hardware,
            camera_stats=cam_stats,
            overall_detection_fps=(
                total_frames / total_det_sec if total_det_sec > 0 else 0.0
            ),
            overall_tracking_fps=(
                total_frames / total_trk_sec if total_trk_sec > 0 else 0.0
            ),
            avg_reid_extraction_ms=(
                (total_reid_sec / total_frames) * 1000.0 if total_frames > 0 else 0.0
            ),
            avg_association_ms=(
                (total_assoc_sec / total_frames) * 1000.0 if total_frames > 0 else 0.0
            ),
            avg_e2e_latency_ms=(
                (total_e2e_sec / total_frames) * 1000.0 if total_frames > 0 else 0.0
            ),
            cpu_percent=cpu_pct,
            peak_memory_mb=round(peak_rss_mb, 2),
            is_real_video=True,
            video_sources=[str(p) for _, p in resolved_sources],
            metrics=metrics,
            failure_cases=failure_cases[:50],
        )

        if results_json_path is not None:
            out_path = Path(results_json_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS-X Multi-Camera Video Benchmark")
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Camera-video pairs in cam_id:path format (minimum 2).",
    )
    parser.add_argument("--ground-truth", default=None, help="Optional GT JSON path.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--output-json",
        default="evaluation/results/latest_benchmark.json",
    )
    args = parser.parse_args()

    cam_videos: list[tuple[str, str]] = []
    for item in args.videos:
        if ":" not in item:
            raise SystemExit(f"Invalid --videos entry '{item}', expected cam_id:path")
        cid, vpath = item.split(":", 1)
        cam_videos.append((cid, vpath))

    bench = MultiCameraBenchmarker(max_frames_per_camera=args.max_frames)
    res = bench.run_on_videos(
        cam_videos,
        ground_truth_path=args.ground_truth,
        results_json_path=args.output_json,
    )
    print(json.dumps(res.to_dict(), indent=2))


if __name__ == "__main__":
    main()
