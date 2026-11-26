"""Single video upload, private streaming, and ARGUS-X analysis routes (/api/videos)."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.security import require_roles, verify_csrf
from backend.config import PROJECT_ROOT
from backend.database.session import SessionLocal, get_db
from backend.models import UploadedVideo, User
from backend.services.audit_service import record_audit_event

if TYPE_CHECKING:
    import cv2
    import numpy as np
    from backend.services.ai_engine import ArgusAIEngineBridge

router = APIRouter(prefix="/api/videos", tags=["videos"])

PRIVATE_VIDEO_DIR = PROJECT_ROOT / "storage" / "uploads" / "videos"
PRIVATE_PROCESSED_DIR = PROJECT_ROOT / "storage" / "processed" / "videos"
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


class AnalyzeVideoRequest(BaseModel):
    max_frames: Optional[int] = Field(default=None, ge=10, le=5000)


def _bridge(request: Request) -> ArgusAIEngineBridge:
    return request.app.state.ai_bridge


def _safe_resolve_private_path(raw_path: str, allowed_root: Path) -> Path:
    """Prevent path traversal by ensuring resolved path stays inside allowed_root."""
    root = allowed_root.resolve()
    target = Path(raw_path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=403, detail="Invalid storage path.")
    return target


def _serialize_video(v: UploadedVideo) -> dict[str, object]:
    results = None
    if v.results_json:
        try:
            results = json.loads(v.results_json)
        except Exception:
            results = None
    return {
        "id": v.id,
        "video_uid": v.video_uid,
        "original_filename": v.original_filename,
        "camera_label": v.camera_label,
        "camera_code": v.camera_code,
        "file_size_bytes": v.file_size_bytes,
        "duration_sec": round(float(v.duration_sec), 2),
        "fps": round(float(v.fps), 2),
        "width": v.width,
        "height": v.height,
        "total_frames": v.total_frames,
        "analysis_status": v.analysis_status,
        "analysis_progress": round(float(v.analysis_progress), 1),
        "has_processed_video": bool(v.processed_path and Path(v.processed_path).is_file()),
        "error_message": v.error_message,
        "results": results,
        "created_at": v.created_at.isoformat(),
        "updated_at": v.updated_at.isoformat(),
    }


def _find_video(db: Session, video_id: str) -> UploadedVideo:
    v = None
    if video_id.isdigit():
        v = db.query(UploadedVideo).filter(UploadedVideo.id == int(video_id)).first()
    if v is None:
        v = db.query(UploadedVideo).filter(UploadedVideo.video_uid == video_id).first()
    if v is None:
        raise HTTPException(status_code=404, detail="Uploaded video not found.")
    return v


def _probe_video_metadata(path: Path) -> tuple[float, float, int, int, int]:
    """Extract (duration_sec, fps, width, height, total_frames) via OpenCV."""
    try:
        import cv2
    except ImportError:
        return 0.0, 25.0, 0, 0, 0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Unable to decode uploaded video file.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if total_frames > 0 else 0.0
        return duration, fps, width, height, total_frames
    finally:
        cap.release()


def _open_video_writer(out_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a browser-friendly VideoWriter (avc1 fallback to mp4v)."""
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, max(1.0, fps), (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Failed to initialize OpenCV VideoWriter for processed output.")


def _analyze_video_worker(
    video_db_id: int,
    max_frames: Optional[int],
    bridge: ArgusAIEngineBridge,
) -> None:
    db = SessionLocal()
    try:
        v = db.query(UploadedVideo).filter(UploadedVideo.id == video_db_id).first()
        if v is None:
            return
        v.analysis_status = "PROCESSING"
        v.analysis_progress = 0.0
        v.error_message = None
        db.commit()

        import cv2
        import numpy as np
        from association import AssociationConfig
        from identity_memory import MultiCameraIdentityManager, TrackletManager
        from tracking.tracker import ObjectTracker

        src_path = Path(v.storage_path)
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {src_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or v.fps or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or v.width or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or v.height or 480)
        total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or v.total_frames or 0)
        if max_frames is not None and total_est > 0:
            total_est = min(total_est, max_frames)

        out_path = PRIVATE_PROCESSED_DIR / f"{v.video_uid}_annotated.mp4"
        writer = _open_video_writer(out_path, fps, width, height)

        tracker = ObjectTracker(device="cpu")
        extractor = bridge._get_extractor()
        cam_id = v.camera_code or v.camera_label or v.video_uid
        tm = TrackletManager(
            camera_id=cam_id,
            feature_extractor=None,
            inactive_timeout_sec=0.3,
            extraction_interval_sec=0.2,
        )
        mc_mgr = MultiCameraIdentityManager(
            db_path=":memory:",
            config=AssociationConfig(max_time_gap_sec=3600.0),
        )

        # Track stats keyed by persistent Global ID (or fallback local track)
        global_stats: dict[str, dict[str, object]] = {}
        local_to_global: dict[int, str] = {}
        recovery_events: list[dict[str, object]] = []
        seen_recoveries: set[tuple[int, str, str]] = set()
        timeline_series: list[dict[str, object]] = []
        total_detections = 0
        frame_idx = 0
        t_proc_total = 0.0

        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                ts = frame_idx / max(1.0, fps)
                t0 = time.perf_counter()
                with bridge._inference_lock:
                    tracked = tracker.track_frame(frame)
                    person_objs = [o for o in tracked if o.class_id == 0]
                    precomputed: dict[int, np.ndarray] = {}
                    if person_objs:
                        crops, tids = [], []
                        for o in person_objs:
                            try:
                                crops.append(extractor.crop_person(frame, o.bbox))
                                tids.append(o.track_id)
                            except ValueError:
                                continue
                        if crops:
                            feats = extractor.extract_batch(crops)
                            for feat_row, tid in zip(feats, tids):
                                precomputed[tid] = feat_row
                t1 = time.perf_counter()
                t_proc_total += (t1 - t0)

                upd = tm.update(tracked, timestamp=ts, precomputed_embeddings=precomputed)
                outs = mc_mgr.process_tracklets(upd.active_tracklets, current_timestamp=ts)

                out_by_local: dict[int, object] = {}
                for out in outs:
                    out_by_local[out.local_track_id] = out
                    if out.global_id:
                        local_to_global[out.local_track_id] = out.global_id
                        rec_key = (out.local_track_id, out.global_id, out.recovery_decision)
                        if out.recovery_decision in ("RECOVERED", "UNCERTAIN") and rec_key not in seen_recoveries:
                            seen_recoveries.add(rec_key)
                            recovery_events.append(
                                {
                                    "frame": frame_idx + 1,
                                    "time_sec": round(ts, 2),
                                    "local_track_id": out.local_track_id,
                                    "global_id": out.global_id,
                                    "decision": out.recovery_decision,
                                    "confidence": round(float(out.confidence), 4),
                                }
                            )

                annotated = frame.copy()
                for obj in person_objs:
                    total_detections += 1
                    tid = int(obj.track_id)
                    out = out_by_local.get(tid)
                    gid = (out.global_id if out and out.global_id else local_to_global.get(tid)) or f"LOCAL-{tid}"
                    decision = out.recovery_decision if out else "TRACKING"
                    is_recovered = decision == "RECOVERED"

                    x1, y1, x2, y2 = [int(c) for c in obj.bbox]
                    box_color = (16, 185, 129) if is_recovered else (56, 189, 248)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{gid} (L:{tid})" + (" RECOVERED" if is_recovered else "")
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2,
                        cv2.LINE_AA,
                    )

                    if gid not in global_stats:
                        global_stats[gid] = {
                            "global_id": gid,
                            "local_track_ids": [tid],
                            "first_frame": frame_idx + 1,
                            "last_frame": frame_idx + 1,
                            "frames_seen": 1,
                            "recovered_count": 1 if is_recovered else 0,
                            "max_confidence": round(float(obj.confidence), 3),
                        }
                    else:
                        st = global_stats[gid]
                        if tid not in st["local_track_ids"]:
                            st["local_track_ids"].append(tid)
                        st["last_frame"] = frame_idx + 1
                        st["frames_seen"] = int(st["frames_seen"]) + 1
                        if is_recovered:
                            st["recovered_count"] = int(st["recovered_count"]) + 1
                        st["max_confidence"] = max(
                            float(st["max_confidence"]), round(float(obj.confidence), 3)
                        )

                writer.write(annotated)
                frame_idx += 1

                if frame_idx % 5 == 0 or frame_idx == 1:
                    timeline_series.append(
                        {
                            "frame": frame_idx,
                            "time_sec": round(ts, 2),
                            "detections": len(person_objs),
                            "active_tracks": len(global_stats),
                        }
                    )

                if total_est > 0 and frame_idx % 10 == 0:
                    v.analysis_progress = min(99.0, (frame_idx / total_est) * 100.0)
                    db.commit()
        finally:
            mc_mgr.close()
            cap.release()
            writer.release()

        avg_fps = frame_idx / t_proc_total if t_proc_total > 0 else 0.0
        avg_latency = (t_proc_total / frame_idx) * 1000.0 if frame_idx > 0 else 0.0
        results_payload = {
            "frames_processed": frame_idx,
            "unique_global_ids": sorted(global_stats.keys()),
            "unique_track_ids": sorted(global_stats.keys()),
            "unique_track_count": len(global_stats),
            "local_track_count": len(local_to_global),
            "reappearance_recoveries": len([e for e in recovery_events if e["decision"] == "RECOVERED"]),
            "recovery_events": recovery_events,
            "total_detections": total_detections,
            "avg_detections_per_frame": round(total_detections / max(1, frame_idx), 2),
            "processing_fps": round(avg_fps, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "tracks": list(global_stats.values()),
            "timeline": timeline_series[-120:],
        }

        v.processed_path = str(out_path)
        v.analysis_status = "COMPLETED"
        v.analysis_progress = 100.0
        v.results_json = json.dumps(results_payload)
        db.commit()
    except Exception as exc:
        try:
            v = db.query(UploadedVideo).filter(UploadedVideo.id == video_db_id).first()
            if v is not None:
                v.analysis_status = "FAILED"
                v.error_message = str(exc)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_single_video(
    request: Request,
    file: UploadFile = File(...),
    camera_label: str = Form("Single Camera Feed"),
    actor: User = Depends(require_roles("ADMIN", "OPERATOR")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Validate and privately store an uploaded video outside the public web root."""
    orig_name = Path(file.filename or "upload.mp4").name
    ext = Path(orig_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTS)}",
        )

    PRIVATE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_uid = f"vid_{uuid.uuid4().hex[:12]}"
    safe_stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(orig_name).stem)[:48] or "video"
    stored_filename = f"{video_uid}_{safe_stem}{ext}"
    dest = _safe_resolve_private_path(str(PRIVATE_VIDEO_DIR / stored_filename), PRIVATE_VIDEO_DIR)

    total_bytes = 0
    with dest.open("wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                out_f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video exceeds 200 MB limit.")
            out_f.write(chunk)

    if total_bytes == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded video file is empty.")

    try:
        duration_sec, fps, width, height, total_frames = _probe_video_metadata(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Invalid video file: {exc}") from exc

    v = UploadedVideo(
        video_uid=video_uid,
        original_filename=orig_name,
        stored_filename=stored_filename,
        storage_path=str(dest),
        camera_label=camera_label.strip() or "Single Camera Feed",
        file_size_bytes=total_bytes,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        analysis_status="UPLOADED",
        analysis_progress=0.0,
        uploaded_by_user_id=actor.id,
    )
    db.add(v)
    db.commit()
    db.refresh(v)

    record_audit_event(
        db,
        action="VIDEO_UPLOADED",
        resource_type="uploaded_video",
        resource_id=v.video_uid,
        details=f"filename={orig_name}, label={v.camera_label}, bytes={total_bytes}",
        actor=actor,
        ip_address=request.client.host if request.client else None,
    )
    return {"video": _serialize_video(v)}


@router.get("")
def list_uploaded_videos(
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    rows = db.query(UploadedVideo).order_by(UploadedVideo.created_at.desc()).limit(100).all()
    return {"videos": [_serialize_video(r) for r in rows], "count": len(rows)}


@router.get("/{video_id}")
def get_uploaded_video(
    video_id: str,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    v = _find_video(db, video_id)
    return {"video": _serialize_video(v)}


@router.post("/{video_id}/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze_uploaded_video(
    video_id: str,
    request: Request,
    payload: Optional[AnalyzeVideoRequest] = None,
    actor: User = Depends(require_roles("ADMIN", "OPERATOR")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Start ARGUS-X tracking + Re-ID analysis on an uploaded video."""
    import os
    if os.getenv("ARGUS_CLOUD_MODE", "").lower() in ("1", "true", "yes"):
        raise HTTPException(
            status_code=503,
            detail="Live AI video analysis (YOLO11 + BoT-SORT + OSNet) is disabled on the lightweight cloud API tier. Run the local backend (http://localhost:5173) for full AI video processing.",
        )

    v = _find_video(db, video_id)
    if v.analysis_status == "PROCESSING":
        raise HTTPException(status_code=409, detail="Video analysis is already in progress.")
    if not Path(v.storage_path).is_file():
        raise HTTPException(
            status_code=404,
            detail="Video file not found on this server. Please re-upload the video here.",
        )

    max_frames = payload.max_frames if payload else None
    v.analysis_status = "QUEUED"
    v.analysis_progress = 0.0
    db.commit()
    db.refresh(v)

    record_audit_event(
        db,
        action="VIDEO_ANALYSIS_STARTED",
        resource_type="uploaded_video",
        resource_id=v.video_uid,
        details=f"max_frames={max_frames}",
        actor=actor,
        ip_address=request.client.host if request.client else None,
    )

    bridge = _bridge(request)
    worker = threading.Thread(
        target=_analyze_video_worker,
        args=(v.id, max_frames, bridge),
        name=f"argus-vid-{v.video_uid}",
        daemon=True,
    )
    worker.start()

    return {"video": _serialize_video(v)}


@router.get("/{video_id}/results")
def get_video_analysis_results(
    video_id: str,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    v = _find_video(db, video_id)
    serialized = _serialize_video(v)
    return {
        "video": serialized,
        "results": serialized["results"],
    }


@router.get("/{video_id}/stream")
def stream_private_video(
    video_id: str,
    kind: str = Query(default="processed"),
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
):
    """Serve private uploaded or processed video only to authenticated users."""
    v = _find_video(db, video_id)
    if kind == "processed" and v.processed_path and Path(v.processed_path).is_file():
        target = _safe_resolve_private_path(v.processed_path, PRIVATE_PROCESSED_DIR)
    else:
        target = _safe_resolve_private_path(v.storage_path, PRIVATE_VIDEO_DIR)

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Video file not found on disk.")
    return FileResponse(path=str(target), media_type="video/mp4", filename=target.name)


@router.delete("/{video_id}")
def delete_uploaded_video(
    video_id: str,
    request: Request,
    actor: User = Depends(require_roles("ADMIN", "OPERATOR")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    v = _find_video(db, video_id)
    for raw_p, root in [(v.storage_path, PRIVATE_VIDEO_DIR), (v.processed_path, PRIVATE_PROCESSED_DIR)]:
        if raw_p:
            try:
                p = _safe_resolve_private_path(raw_p, root)
                p.unlink(missing_ok=True)
            except Exception:
                pass
    uid = v.video_uid
    db.delete(v)
    db.commit()
    record_audit_event(
        db,
        action="VIDEO_DELETED",
        resource_type="uploaded_video",
        resource_id=uid,
        actor=actor,
        ip_address=request.client.host if request.client else None,
    )
    return {"message": "Video deleted."}
