"""End-to-end multi-camera identity recovery demonstration for ARGUS-X.

Reuses the existing CameraManager, ObjectTracker (YOLO11n + BoT-SORT),
PersonFeatureExtractor (OSNet x0.25), IdentityMemory (FAISS + SQLite),
DataAssociator (Hungarian), MultiCameraIdentityManager, and evaluation metrics
without duplicating models or exposing raw biometric embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import psutil

from api.state import ArgusServiceState
from association import CameraTransitionRule
from config import BASE_DIR
from detection.detector import Detector
from evaluation.benchmark import detect_hardware_info
from evaluation.metrics import (
    EvaluationMetrics,
    GroundTruthAnnotation,
    PredictedObservation,
    evaluate_predictions,
    load_ground_truth,
)
from reidentification.feature_extractor import PersonFeatureExtractor
from tracking.tracker import ObjectTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemoSummary:
    """Structured summary of an end-to-end multi-camera demonstration run."""

    videos_processed: int
    video_sources: list[str]
    is_synthetic_footage: bool
    total_frames_processed: int
    total_people_detections: int
    unique_global_identities: int
    identity_recoveries: int
    uncertain_decisions: int
    rejected_decisions: int
    conflicts_blocked: int
    overall_fps: float
    avg_e2e_latency_ms: float
    per_camera_time_sec: dict[str, float]
    cpu_percent: float
    peak_memory_mb: float
    active_device: str
    verified_against_ground_truth: bool
    metrics: Optional[EvaluationMetrics] = None
    audit_log_count: int = 0
    unresolved_pending_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def create_sample_demo_videos(
    output_dir: Union[str, Path],
    source_image: Union[str, Path] = BASE_DIR / "bus.jpg",
) -> tuple[list[tuple[str, Path]], Path]:
    """Build two short synthetic MP4 camera clips from authorized sample image crops.

    Clearly labeled as synthetic test footage (not real-world CCTV):
    - cam_1 (t=0.0s..1.0s): Person A + Person B
    - cam_2 (t=7.0s..8.0s): Person A reappears (valid cam_1->cam_2 travel >= 5s)
      alongside Person C (new identity)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(source_image))
    if img is None:
        raise FileNotFoundError(f"Sample image not found: {source_image}")

    det = Detector(device="auto")
    dets = [d for d in det.detect(img) if d.class_id == 0]
    if len(dets) < 2:
        raise RuntimeError("Need at least 2 person detections in sample image.")

    crops: list[np.ndarray] = []
    for d in dets[:3]:
        x1, y1, x2, y2 = [max(0, int(round(v))) for v in d.bbox]
        crops.append(img[y1:y2, x1:x2].copy())
    while len(crops) < 3:
        crops.append(cv2.flip(crops[0], 1))

    fps = 5.0
    width, height = 640, 480
    cam1_path = out_dir / "demo_cam_1.mp4"
    cam2_path = out_dir / "demo_cam_2.mp4"

    def _paste(bg: np.ndarray, crop: np.ndarray, x: int, y: int) -> tuple[float, float, float, float]:
        ch, cw = crop.shape[:2]
        scale = min(220.0 / max(1, ch), 140.0 / max(1, cw))
        nh, nw = max(40, int(ch * scale)), max(24, int(cw * scale))
        resized = cv2.resize(crop, (nw, nh))
        x2, y2 = min(width, x + nw), min(height, y + nh)
        bg[y:y2, x:x2] = resized[: y2 - y, : x2 - x]
        return (float(x), float(y), float(x2), float(y2))

    gt_annotations: list[dict[str, object]] = []

    w1 = cv2.VideoWriter(str(cam1_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(5):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        b_a = _paste(frame, crops[0], 80 + i * 6, 120)
        b_b = _paste(frame, crops[1], 360 + i * 4, 120)
        w1.write(frame)
        ts = i / fps
        gt_annotations.append(
            {"camera_id": "cam_1", "timestamp": ts, "gt_identity_id": "person_A", "bbox": list(b_a), "scenario_tag": "camera_transition"}
        )
        gt_annotations.append(
            {"camera_id": "cam_1", "timestamp": ts, "gt_identity_id": "person_B", "bbox": list(b_b), "scenario_tag": "camera_transition"}
        )
    w1.release()

    w2 = cv2.VideoWriter(str(cam2_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(5):
        frame = np.full((height, width, 3), 45, dtype=np.uint8)
        # cam_2 starts at offset t=7.0s (valid 6s travel after cam_1 ends at 0.8s)
        b_a = _paste(frame, crops[0], 140 + i * 6, 130)
        b_c = _paste(frame, crops[2], 400 + i * 5, 130)
        w2.write(frame)
        ts = 7.0 + i / fps
        gt_annotations.append(
            {"camera_id": "cam_2", "timestamp": ts, "gt_identity_id": "person_A", "bbox": list(b_a), "scenario_tag": "camera_transition"}
        )
        gt_annotations.append(
            {"camera_id": "cam_2", "timestamp": ts, "gt_identity_id": "person_C", "bbox": list(b_c), "scenario_tag": "different_person"}
        )
    w2.release()

    gt_path = out_dir / "demo_ground_truth.json"
    gt_path.write_text(json.dumps({"annotations": gt_annotations}, indent=2), encoding="utf-8")
    return [("cam_1", cam1_path), ("cam_2", cam2_path)], gt_path


class MultiCameraDemoRunner:
    """Executes the full ARGUS-X multi-camera pipeline on local video files."""

    def __init__(
        self,
        transition_rules: Optional[Sequence[CameraTransitionRule]] = None,
        device: str = "auto",
        db_path: Union[str, Path] = ":memory:",
        index_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.hardware = detect_hardware_info(device)
        self.rules = list(
            transition_rules
            or [
                CameraTransitionRule("cam_1", "cam_1", 0.0, 300.0),
                CameraTransitionRule("cam_2", "cam_2", 0.0, 300.0),
                CameraTransitionRule("cam_1", "cam_2", 5.0, 120.0),
                CameraTransitionRule("cam_2", "cam_1", 5.0, 120.0),
            ]
        )
        self.service = ArgusServiceState(
            db_path=db_path,
            index_path=index_path,
            transition_rules=self.rules,
        )
        # Share a single OSNet feature extractor across all cameras.
        self.extractor = PersonFeatureExtractor(device=self.hardware.active_device)
        self.service.camera_manager.feature_extractor = self.extractor

    def run(
        self,
        camera_videos: Sequence[tuple[str, Union[str, Path]]],
        camera_start_offsets_sec: Optional[dict[str, float]] = None,
        ground_truth_path: Optional[Union[str, Path]] = None,
        is_synthetic_footage: bool = False,
        max_frames_per_camera: Optional[int] = None,
    ) -> DemoSummary:
        if len(camera_videos) < 2:
            raise ValueError("Demonstration requires at least 2 local video files.")

        offsets = camera_start_offsets_sec or {}
        proc = psutil.Process(os.getpid())
        proc.cpu_percent(interval=None)
        peak_rss_mb = proc.memory_info().rss / (1024.0 ** 2)

        for cam_id, vpath in camera_videos:
            tracker = ObjectTracker(device=self.hardware.active_device)
            self.service.camera_manager.register_camera(
                config=__import__("camera").CameraConfig(camera_id=cam_id, source=str(vpath)),
                tracker=tracker,
            )
            self.service.camera_manager.start_camera(cam_id)

        predictions: list[PredictedObservation] = []
        per_cam_time: dict[str, float] = {}
        total_frames = 0
        total_people_detections = 0
        t_all_start = time.perf_counter()

        try:
            for cam_id, vpath in camera_videos:
                state = self.service.camera_manager.get_camera(cam_id)
                cap = state.capture
                assert cap is not None
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                base_offset = float(offsets.get(cam_id, 0.0))

                c_frames = 0
                c_start = time.perf_counter()
                while True:
                    if max_frames_per_camera is not None and c_frames >= max_frames_per_camera:
                        break
                    frame = self.service.camera_manager.read_frame(cam_id)
                    if frame is None:
                        break

                    ts = base_offset + (c_frames / fps)
                    upd = self.service.camera_manager.process_camera_frame(
                        cam_id, frame, timestamp=ts
                    )
                    total_people_detections += sum(
                        len(t.samples) for t in upd.newly_started
                    ) + len(upd.active_tracklets)

                    all_active = []
                    for cid in self.service.camera_manager.list_camera_ids():
                        all_active.extend(
                            self.service.camera_manager.get_camera(cid).tracklet_manager.get_active_tracklets()
                        )
                    outs = self.service.identity_manager.process_tracklets(
                        all_active, current_timestamp=ts
                    )
                    cam_outs = [o for o in outs if o.camera_id == cam_id]
                    self.service.ingest_frame_outputs(cam_id, frame, cam_outs, ts)

                    for out in cam_outs:
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

                    c_frames += 1
                    rss = proc.memory_info().rss / (1024.0 ** 2)
                    if rss > peak_rss_mb:
                        peak_rss_mb = rss

                per_cam_time[cam_id] = round(time.perf_counter() - c_start, 4)
                total_frames += c_frames
                # Disconnect camera cleanly after its stream ends so identities
                # become TEMPORARILY_LOST and eligible for recovery in next camera.
                self.service.disconnect_camera(cam_id)
        finally:
            total_elapsed = max(1e-6, time.perf_counter() - t_all_start)

        cpu_pct = proc.cpu_percent(interval=None)
        audit_logs = self.service.identity_manager.get_audit_logs()
        recoveries = sum(1 for a in audit_logs if a.decision == "RECOVERED_ACCEPT")
        uncertains = sum(1 for a in audit_logs if a.decision == "UNCERTAIN")
        rejects = sum(1 for a in audit_logs if a.decision == "REJECT_NEW_IDENTITY")
        conflicts = len(self.service.identity_manager.get_conflict_events())

        metrics: Optional[EvaluationMetrics] = None
        verified = False
        if ground_truth_path is not None:
            gt = load_ground_truth(ground_truth_path)
            metrics = evaluate_predictions(gt, predictions)
            verified = True

        return DemoSummary(
            videos_processed=len(camera_videos),
            video_sources=[str(p) for _, p in camera_videos],
            is_synthetic_footage=is_synthetic_footage,
            total_frames_processed=total_frames,
            total_people_detections=total_people_detections,
            unique_global_identities=len(self.service.identity_manager.list_identities()),
            identity_recoveries=recoveries,
            uncertain_decisions=uncertains,
            rejected_decisions=rejects,
            conflicts_blocked=conflicts,
            overall_fps=round(total_frames / total_elapsed, 2),
            avg_e2e_latency_ms=round((total_elapsed / max(1, total_frames)) * 1000.0, 2),
            per_camera_time_sec=per_cam_time,
            cpu_percent=cpu_pct,
            peak_memory_mb=round(peak_rss_mb, 2),
            active_device=self.hardware.active_device,
            verified_against_ground_truth=verified,
            metrics=metrics,
            audit_log_count=len(audit_logs),
            unresolved_pending_count=len(
                self.service.identity_manager.list_unresolved_observations(status="PENDING")
            ),
        )

    def close(self) -> None:
        self.service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ARGUS-X Multi-Camera End-to-End Demo")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Camera-video pairs in cam_id:path format.",
    )
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--generate-sample-demo", action="store_true")
    parser.add_argument("--sample-dir", default="evaluation/results/sample_demo")
    args = parser.parse_args()

    if args.generate_sample_demo or not args.videos:
        cam_videos, gt_path = create_sample_demo_videos(args.sample_dir)
        offsets = {"cam_1": 0.0, "cam_2": 7.0}
        is_synth = True
    else:
        cam_videos = [tuple(item.split(":", 1)) for item in args.videos]  # type: ignore[misc]
        gt_path = args.ground_truth
        offsets = {}
        is_synth = False

    runner = MultiCameraDemoRunner()
    try:
        summary = runner.run(
            cam_videos,
            camera_start_offsets_sec=offsets,
            ground_truth_path=gt_path,
            is_synthetic_footage=is_synth,
        )
        print(json.dumps(summary.to_dict(), indent=2))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
