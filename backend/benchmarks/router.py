"""MOT17 benchmarking API routes (/api/benchmark)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.security import require_roles, verify_csrf
from backend.database.session import SessionLocal, get_db
from backend.datasets.service import (
    discover_mot17_sequences,
    load_mot17_ground_truth,
    sync_mot17_dataset_record,
)
from backend.benchmarks.mot_eval import evaluate_mot17_predictions
from backend.models import BenchmarkJob, BenchmarkResult, User
from backend.services.ai_engine import ArgusAIEngineBridge
from backend.services.audit_service import record_audit_event
from tracking.tracker import ObjectTracker

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


class MOT17BenchmarkRequest(BaseModel):
    sequence_name: str = Field(..., min_length=3, max_length=64)
    max_frames: Optional[int] = Field(default=60, ge=5, le=2000)


def _bridge(request: Request) -> ArgusAIEngineBridge:
    return request.app.state.ai_bridge


def _serialize_job(job: BenchmarkJob) -> dict[str, object]:
    return {
        "id": job.id,
        "job_uid": job.job_uid,
        "dataset_name": job.dataset_name,
        "sequence_name": job.sequence_name,
        "detector_type": job.detector_type,
        "status": job.status,
        "progress": round(float(job.progress), 1),
        "processed_frames": job.processed_frames,
        "total_frames": job.total_frames,
        "max_frames": job.max_frames,
        "error_message": job.error_message,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "created_at": job.created_at.isoformat(),
    }


def _serialize_result(res: BenchmarkResult) -> dict[str, object]:
    details = {}
    try:
        details = json.loads(res.details_json)
    except Exception:
        details = {}
    return {
        "id": res.id,
        "job_id": res.job_id,
        "sequence_name": res.sequence_name,
        "mota": res.mota,
        "idf1": res.idf1,
        "hota": res.hota,
        "id_switches": res.id_switches,
        "false_positives": res.false_positives,
        "false_negatives": res.false_negatives,
        "true_positives": res.true_positives,
        "total_gt": res.total_gt,
        "fps": res.fps,
        "processing_latency_ms": res.processing_latency_ms,
        "frames_evaluated": res.frames_evaluated,
        "details": details,
        "created_at": res.created_at.isoformat(),
    }


def _find_job(db: Session, job_id: str) -> BenchmarkJob:
    job = None
    if job_id.isdigit():
        job = db.query(BenchmarkJob).filter(BenchmarkJob.id == int(job_id)).first()
    if job is None:
        job = db.query(BenchmarkJob).filter(BenchmarkJob.job_uid == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Benchmark job not found.")
    return job


def _run_mot17_benchmark_worker(
    job_id: int,
    seq_info: dict[str, object],
    max_frames: Optional[int],
    bridge: ArgusAIEngineBridge,
) -> None:
    db = SessionLocal()
    try:
        job = db.query(BenchmarkJob).filter(BenchmarkJob.id == job_id).first()
        if job is None:
            return
        job.status = "RUNNING"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        seq_dir = Path(str(seq_info["sequence_dir"]))
        img_dir = seq_dir / str(seq_info["im_dir"])
        im_ext = str(seq_info["im_ext"])
        gt_path = seq_dir / "gt" / "gt.txt"

        frame_files = sorted(img_dir.glob(f"*{im_ext}"))
        if max_frames is not None and max_frames > 0:
            frame_files = frame_files[:max_frames]
        total_to_eval = len(frame_files)
        job.total_frames = total_to_eval
        db.commit()

        if total_to_eval == 0:
            raise RuntimeError(f"No {im_ext} frames found in {img_dir}")

        gt_by_frame = load_mot17_ground_truth(gt_path, pedestrian_only=True)
        tracker = ObjectTracker(device="cpu")

        pred_by_frame: dict[int, list[dict[str, float]]] = {}
        evaluated_frames: list[int] = []
        total_proc_sec = 0.0

        for idx, frame_path in enumerate(frame_files, start=1):
            # MOT17 filenames are 000001.jpg -> 1-based frame index
            try:
                frame_num = int(frame_path.stem)
            except ValueError:
                frame_num = idx

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            t0 = time.perf_counter()
            with bridge._inference_lock:
                tracked = tracker.track_frame(frame)
            t1 = time.perf_counter()
            total_proc_sec += (t1 - t0)

            frame_preds: list[dict[str, float]] = []
            for obj in tracked:
                if obj.class_id != 0:
                    continue
                x1, y1, x2, y2 = obj.bbox
                frame_preds.append(
                    {
                        "id": int(obj.track_id),
                        "x": float(x1),
                        "y": float(y1),
                        "w": float(max(0.0, x2 - x1)),
                        "h": float(max(0.0, y2 - y1)),
                    }
                )
            pred_by_frame[frame_num] = frame_preds
            evaluated_frames.append(frame_num)

            if idx % 5 == 0 or idx == total_to_eval:
                job.processed_frames = idx
                job.progress = (idx / total_to_eval) * 100.0
                db.commit()

        metrics = evaluate_mot17_predictions(
            gt_by_frame=gt_by_frame,
            pred_by_frame=pred_by_frame,
            evaluated_frames=evaluated_frames,
        )
        fps = len(evaluated_frames) / total_proc_sec if total_proc_sec > 0 else 0.0
        latency_ms = (total_proc_sec / len(evaluated_frames)) * 1000.0 if evaluated_frames else 0.0

        res = BenchmarkResult(
            job_id=job.id,
            sequence_name=job.sequence_name,
            mota=metrics.mota,
            idf1=metrics.idf1,
            hota=metrics.hota,
            id_switches=metrics.id_switches,
            false_positives=metrics.false_positives,
            false_negatives=metrics.false_negatives,
            true_positives=metrics.true_positives,
            total_gt=metrics.total_gt,
            fps=round(fps, 2),
            processing_latency_ms=round(latency_ms, 2),
            frames_evaluated=metrics.frames_evaluated,
            details_json=json.dumps(
                {
                    "detector_type": job.detector_type,
                    "total_processing_sec": round(total_proc_sec, 3),
                    "image_width": seq_info.get("im_width"),
                    "image_height": seq_info.get("im_height"),
                    "sequence_frame_rate": seq_info.get("frame_rate"),
                }
            ),
        )
        db.add(res)
        job.status = "COMPLETED"
        job.progress = 100.0
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        try:
            job = db.query(BenchmarkJob).filter(BenchmarkJob.id == job_id).first()
            if job is not None:
                job.status = "FAILED"
                job.error_message = str(exc)
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.get("")
def list_benchmark_jobs(
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """List recent MOT17 benchmark jobs and their results if completed."""
    jobs = db.query(BenchmarkJob).order_by(BenchmarkJob.created_at.desc()).limit(50).all()
    items = []
    for job in jobs:
        d = _serialize_job(job)
        if job.result is not None:
            d["result"] = _serialize_result(job.result)
        items.append(d)
    return {"jobs": items, "count": len(items)}


@router.post("/mot17", status_code=status.HTTP_202_ACCEPTED)
def start_mot17_benchmark(
    payload: MOT17BenchmarkRequest,
    request: Request,
    actor: User = Depends(require_roles("ADMIN", "OPERATOR")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Start an asynchronous MOT17 benchmark job using YOLO11n + BoT-SORT."""
    ds_info = sync_mot17_dataset_record(db)
    if ds_info["status"] != "AVAILABLE":
        raise HTTPException(
            status_code=404,
            detail=ds_info.get("setup_message") or "MOT17 dataset is not available.",
        )

    all_info = discover_mot17_sequences(detector_filter=None, split_filter="train")
    target_seq = None
    for seq in all_info["sequences"]:
        if seq["name"] == payload.sequence_name.strip():
            target_seq = seq
            break

    if target_seq is None:
        raise HTTPException(
            status_code=404,
            detail=f"MOT17 sequence '{payload.sequence_name}' not found in train/.",
        )
    if not target_seq["is_valid"]:
        raise HTTPException(
            status_code=422,
            detail=f"Sequence '{payload.sequence_name}' failed validation: {target_seq['validation_errors']}",
        )

    total_frames = int(target_seq["actual_frame_count"])
    if payload.max_frames is not None:
        total_frames = min(total_frames, payload.max_frames)

    job = BenchmarkJob(
        job_uid=f"bench_{uuid.uuid4().hex[:12]}",
        dataset_id=ds_info.get("dataset_id"),
        dataset_name="MOT17",
        sequence_name=str(target_seq["name"]),
        detector_type=str(target_seq["detector_type"]),
        status="QUEUED",
        progress=0.0,
        processed_frames=0,
        total_frames=total_frames,
        max_frames=payload.max_frames,
        created_by_user_id=actor.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    record_audit_event(
        db,
        action="BENCHMARK_STARTED",
        resource_type="benchmark_job",
        resource_id=job.job_uid,
        details=f"sequence={job.sequence_name}, max_frames={job.max_frames}",
        actor=actor,
        ip_address=request.client.host if request.client else None,
    )

    bridge = _bridge(request)
    worker = threading.Thread(
        target=_run_mot17_benchmark_worker,
        args=(job.id, target_seq, payload.max_frames, bridge),
        name=f"argus-bench-{job.job_uid}",
        daemon=True,
    )
    worker.start()

    return {"job": _serialize_job(job)}


@router.get("/{job_id}")
def get_benchmark_job(
    job_id: str,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Return status and progress of a benchmark job."""
    job = _find_job(db, job_id)
    return {"job": _serialize_job(job)}


@router.get("/{job_id}/results")
def get_benchmark_results(
    job_id: str,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Return full TrackEval-compatible benchmark results for a job."""
    job = _find_job(db, job_id)
    if job.status != "COMPLETED" or job.result is None:
        return {
            "job": _serialize_job(job),
            "result": None,
        }
    return {
        "job": _serialize_job(job),
        "result": _serialize_result(job.result),
    }
