"""Multi-camera video analysis & persistent identity recovery routes (/api/multicamera)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.security import require_roles, verify_csrf
from backend.database.session import SessionLocal, get_db
from backend.models import (
    Camera,
    Identity,
    IdentityObservation,
    RecoveryEvent,
    UploadedVideo,
    User,
)
from backend.services.audit_service import record_audit_event

if TYPE_CHECKING:
    import numpy as np
    from backend.services.ai_engine import ArgusAIEngineBridge

router = APIRouter(prefix="/api/multicamera", tags=["multicamera"])

# In-memory registry of multi-camera analysis sessions (no raw embeddings stored)
_MC_SESSIONS_LOCK = threading.Lock()
_MC_SESSIONS: dict[str, dict[str, object]] = {}


class CameraVideoAssignment(BaseModel):
    video_uid: str = Field(..., min_length=3, max_length=64)
    camera_code: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    camera_name: Optional[str] = Field(default=None, max_length=128)


class MultiCameraAnalyzeRequest(BaseModel):
    cameras: list[CameraVideoAssignment] = Field(..., min_length=2, max_length=8)
    max_frames_per_camera: Optional[int] = Field(default=90, ge=5, le=2000)


def _bridge(request: Request) -> ArgusAIEngineBridge:
    return request.app.state.ai_bridge


def _sync_identity_to_neon(
    db: Session,
    global_id: str,
    lifecycle_state: str,
    camera_code: str,
    local_track_id: int,
    confidence: float,
    decision: str,
    reason: str,
    from_camera: Optional[str] = None,
) -> None:
    """Persist identity, observation, and recovery event in Neon without raw embeddings."""
    now = datetime.now(timezone.utc)
    ident = db.query(Identity).filter(Identity.global_id == global_id).first()
    if ident is None:
        ident = Identity(
            global_id=global_id,
            lifecycle_state=lifecycle_state,
            current_camera_code=camera_code,
            confidence=confidence,
            has_conflict=(decision == "UNCERTAIN"),
            first_seen_at=now,
            last_seen_at=now,
        )
        db.add(ident)
        db.flush()
    else:
        ident.lifecycle_state = lifecycle_state
        ident.current_camera_code = camera_code
        ident.confidence = max(float(ident.confidence), float(confidence))
        if decision == "UNCERTAIN":
            ident.has_conflict = True
        ident.last_seen_at = now

    obs = IdentityObservation(
        identity_id=ident.id,
        camera_code=camera_code,
        local_track_id=local_track_id,
        confidence=confidence,
        status="UNCERTAIN" if decision == "UNCERTAIN" else "CONFIRMED",
        observed_at=now,
    )
    db.add(obs)

    if decision in ("RECOVERED", "UNCERTAIN", "NEW_IDENTITY"):
        ev = RecoveryEvent(
            identity_id=ident.id,
            global_id=global_id,
            from_camera_code=from_camera,
            to_camera_code=camera_code,
            local_track_id=local_track_id,
            decision=decision,
            confidence=confidence,
            review_status="PENDING" if decision == "UNCERTAIN" else "AUTO",
            reason=reason[:500],
            recovered_at=now,
        )
        db.add(ev)


def _run_multicamera_worker(
    session_uid: str,
    assignments: list[dict[str, object]],
    max_frames: Optional[int],
    bridge: ArgusAIEngineBridge,
) -> None:
    db = SessionLocal()
    try:
        with _MC_SESSIONS_LOCK:
            _MC_SESSIONS[session_uid]["status"] = "PROCESSING"

        extractor = bridge._get_extractor()
        camera_summaries: list[dict[str, object]] = []
        recovery_log: list[dict[str, object]] = []
        seen_track_decisions: dict[tuple[str, int], dict[str, object]] = {}

        total_cams = len(assignments)
        for cam_idx, item in enumerate(assignments):
            cam_code = str(item["camera_code"])
            cam_name = str(item["camera_name"])
            video_path = Path(str(item["storage_path"]))
            video_uid = str(item["video_uid"])

            # Register camera in Neon + CameraManager
            cam_row = db.query(Camera).filter(Camera.camera_code == cam_code).first()
            if cam_row is None:
                cam_row = Camera(
                    camera_code=cam_code,
                    name=cam_name,
                    source=str(video_path),
                    location=f"Multi-Camera Session {session_uid}",
                    enabled=True,
                    status="ONLINE",
                )
                db.add(cam_row)
                db.commit()
                db.refresh(cam_row)
            else:
                cam_row.source = str(video_path)
                cam_row.status = "ONLINE"
                db.commit()

            try:
                bridge.state.camera_manager.get_camera(cam_code)
            except KeyError:
                bridge.state.register_camera(
                    camera_id=cam_code,
                    source=str(video_path),
                    name=cam_name,
                    enabled=True,
                    auto_start=False,
                )

            import cv2
            import numpy as np
            from identity_memory import TrackletManager
            from tracking.tracker import ObjectTracker

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for camera {cam_code}: {video_path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            tracker = ObjectTracker(device="cpu")
            tm = TrackletManager(
                camera_id=cam_code,
                feature_extractor=None,
                inactive_timeout_sec=1.0,
                extraction_interval_sec=0.5,
            )

            cam_local_tracks: set[int] = set()
            cam_global_ids: set[str] = set()
            frames_done = 0
            base_ts = time.time() + (cam_idx * 3.0)

            try:
                while True:
                    if max_frames is not None and frames_done >= max_frames:
                        break
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    ts = base_ts + (frames_done / max(1.0, fps))
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

                    upd = tm.update(tracked, timestamp=ts, precomputed_embeddings=precomputed)
                    with bridge.state._lock:
                        outs = bridge.state.identity_manager.process_tracklets(
                            upd.active_tracklets, current_timestamp=ts
                        )
                        bridge.state.ingest_frame_outputs(cam_code, frame, outs, ts)

                    for out in outs:
                        cam_local_tracks.add(out.local_track_id)
                        if out.global_id:
                            cam_global_ids.add(out.global_id)

                        key = (cam_code, out.local_track_id)
                        prev_entry = seen_track_decisions.get(key)
                        if prev_entry is None or prev_entry["decision"] != out.recovery_decision:
                            entry = {
                                "camera_code": cam_code,
                                "local_track_id": out.local_track_id,
                                "global_id": out.global_id,
                                "decision": out.recovery_decision,
                                "confidence": round(float(out.confidence), 4),
                                "reason": out.reason,
                                "timestamp": ts,
                            }
                            seen_track_decisions[key] = entry
                            recovery_log.append(entry)

                            if out.global_id:
                                _sync_identity_to_neon(
                                    db=db,
                                    global_id=out.global_id,
                                    lifecycle_state=out.identity_state.value,
                                    camera_code=cam_code,
                                    local_track_id=out.local_track_id,
                                    confidence=float(out.confidence),
                                    decision=out.recovery_decision,
                                    reason=out.reason,
                                )
                    frames_done += 1
                    if frames_done % 10 == 0:
                        db.commit()
                        with _MC_SESSIONS_LOCK:
                            prog = ((cam_idx + (frames_done / max(1, max_frames or 100))) / total_cams) * 100.0
                            _MC_SESSIONS[session_uid]["progress"] = min(98.0, round(prog, 1))
            finally:
                cap.release()
                try:
                    bridge.state.identity_manager.handle_camera_disconnect(
                        cam_code, timestamp=base_ts + (frames_done / max(1.0, fps))
                    )
                except Exception:
                    pass
                cam_row.status = "OFFLINE"
                db.commit()

            # Update UploadedVideo status
            v_row = db.query(UploadedVideo).filter(UploadedVideo.video_uid == video_uid).first()
            if v_row is not None:
                v_row.camera_code = cam_code
                v_row.analysis_status = "COMPLETED"
                v_row.analysis_progress = 100.0
                db.commit()

            camera_summaries.append(
                {
                    "camera_code": cam_code,
                    "camera_name": cam_name,
                    "video_uid": video_uid,
                    "frames_processed": frames_done,
                    "local_tracks_count": len(cam_local_tracks),
                    "local_track_ids": sorted(cam_local_tracks),
                    "global_ids": sorted(cam_global_ids),
                }
            )

        # Gather accepted, uncertain, rejected counts from Neon + session
        accepted_matches = [r for r in recovery_log if r["decision"] == "RECOVERED"]
        uncertain_matches = [r for r in recovery_log if r["decision"] == "UNCERTAIN"]
        new_identities = [r for r in recovery_log if r["decision"] == "NEW_IDENTITY"]

        rejected_count = db.query(RecoveryEvent).filter(RecoveryEvent.review_status == "REJECTED").count()

        with _MC_SESSIONS_LOCK:
            _MC_SESSIONS[session_uid].update(
                {
                    "status": "COMPLETED",
                    "progress": 100.0,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "camera_summaries": camera_summaries,
                    "recovery_events": recovery_log,
                    "counts": {
                        "cameras": len(camera_summaries),
                        "accepted_matches": len(accepted_matches),
                        "uncertain_matches": len(uncertain_matches),
                        "new_identities": len(new_identities),
                        "rejected_matches": rejected_count,
                    },
                }
            )
    except Exception as exc:
        with _MC_SESSIONS_LOCK:
            if session_uid in _MC_SESSIONS:
                _MC_SESSIONS[session_uid]["status"] = "FAILED"
                _MC_SESSIONS[session_uid]["error_message"] = str(exc)
    finally:
        db.close()


@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
def start_multicamera_analysis(
    payload: MultiCameraAnalyzeRequest,
    request: Request,
    actor: User = Depends(require_roles("ADMIN", "OPERATOR")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Start multi-camera tracking & persistent identity recovery across uploaded videos."""
    assignments: list[dict[str, object]] = []
    seen_codes: set[str] = set()
    for cam in payload.cameras:
        code = cam.camera_code.strip()
        if code in seen_codes:
            raise HTTPException(status_code=422, detail=f"Duplicate camera_code: '{code}'")
        seen_codes.add(code)

        v = db.query(UploadedVideo).filter(UploadedVideo.video_uid == cam.video_uid.strip()).first()
        if v is None:
            raise HTTPException(status_code=404, detail=f"Uploaded video '{cam.video_uid}' not found.")
        if not Path(v.storage_path).is_file():
            raise HTTPException(status_code=404, detail=f"Video file missing for '{cam.video_uid}'.")

        assignments.append(
            {
                "video_uid": v.video_uid,
                "camera_code": code,
                "camera_name": (cam.camera_name or v.camera_label or code).strip(),
                "storage_path": v.storage_path,
            }
        )

    session_uid = f"mc_{uuid.uuid4().hex[:12]}"
    session_obj = {
        "session_uid": session_uid,
        "status": "QUEUED",
        "progress": 0.0,
        "cameras": [
            {"camera_code": a["camera_code"], "camera_name": a["camera_name"], "video_uid": a["video_uid"]}
            for a in assignments
        ],
        "camera_summaries": [],
        "recovery_events": [],
        "counts": {
            "cameras": len(assignments),
            "accepted_matches": 0,
            "uncertain_matches": 0,
            "new_identities": 0,
            "rejected_matches": 0,
        },
        "error_message": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    with _MC_SESSIONS_LOCK:
        _MC_SESSIONS[session_uid] = session_obj

    record_audit_event(
        db,
        action="MULTICAMERA_ANALYSIS_STARTED",
        resource_type="multicamera_session",
        resource_id=session_uid,
        details=json.dumps([a["camera_code"] for a in assignments]),
        actor=actor,
        ip_address=request.client.host if request.client else None,
    )

    bridge = _bridge(request)
    worker = threading.Thread(
        target=_run_multicamera_worker,
        args=(session_uid, assignments, payload.max_frames_per_camera, bridge),
        name=f"argus-mc-{session_uid}",
        daemon=True,
    )
    worker.start()

    return {"session": session_obj}


@router.get("/sessions")
def list_multicamera_sessions(
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
) -> dict[str, object]:
    with _MC_SESSIONS_LOCK:
        items = list(_MC_SESSIONS.values())
    items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return {"sessions": items, "count": len(items)}


@router.get("/sessions/{session_uid}")
def get_multicamera_session(
    session_uid: str,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
) -> dict[str, object]:
    with _MC_SESSIONS_LOCK:
        sess = _MC_SESSIONS.get(session_uid)
    if sess is None:
        raise HTTPException(status_code=404, detail="Multi-camera session not found.")
    return {"session": sess}
