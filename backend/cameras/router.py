"""Camera management routes (/api/cameras) integrated with CameraManager."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from sqlalchemy.orm import Session

from backend.auth.security import require_roles, verify_csrf
from backend.config import PROJECT_ROOT
from backend.database.session import get_db
from backend.models import Camera, User
from backend.schemas import CameraCreateRequest, CameraUpdateRequest
from backend.services.audit_service import record_audit_event

if TYPE_CHECKING:
    from backend.services.ai_engine import ArgusAIEngineBridge

router = APIRouter(prefix="/api/cameras", tags=["cameras"])
UPLOAD_DIR = PROJECT_ROOT / "datasets" / "uploads"
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def _bridge(request: Request) -> ArgusAIEngineBridge:
    return request.app.state.ai_bridge


def _serialize_camera(cam: Camera, bridge: ArgusAIEngineBridge) -> dict[str, object]:
    live_info = None
    try:
        live_info = bridge.state.get_camera_stream_payload(cam.camera_code)
    except KeyError:
        live_info = None

    is_running = bool(live_info["is_running"]) if live_info else False
    return {
        "id": cam.id,
        "camera_code": cam.camera_code,
        "name": cam.name,
        "source": cam.source,
        "location": cam.location,
        "fps": cam.fps,
        "enabled": cam.enabled,
        "status": "ONLINE" if is_running else cam.status,
        "is_running": is_running,
        "active_tracks": len(live_info.get("detections", [])) if live_info else 0,
        "created_at": cam.created_at,
        "updated_at": cam.updated_at,
    }


@router.get("")
def list_cameras(
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    bridge = _bridge(request)
    cams = db.query(Camera).order_by(Camera.id.asc()).all()
    return {"cameras": [_serialize_camera(c, bridge) for c in cams], "count": len(cams)}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreateRequest,
    request: Request,
    admin: User = Depends(require_roles("ADMIN")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    code = payload.camera_code.strip()
    if db.query(Camera).filter(Camera.camera_code == code).first() is not None:
        raise HTTPException(status_code=409, detail=f"Camera '{code}' already exists.")

    cam = Camera(
        camera_code=code,
        name=payload.name.strip(),
        source=str(payload.source) if payload.source is not None else None,
        location=payload.location.strip(),
        fps=payload.fps,
        enabled=payload.enabled,
        status="ONLINE" if payload.auto_start else "OFFLINE",
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    bridge = _bridge(request)
    try:
        bridge.register_or_sync_camera(cam, auto_start=payload.auto_start)
    except Exception as exc:
        cam.status = "ERROR"
        db.commit()
        raise HTTPException(status_code=422, detail=f"Camera AI registration failed: {exc}") from exc

    record_audit_event(
        db,
        action="CAMERA_CREATED",
        resource_type="camera",
        resource_id=str(cam.id),
        details=f"camera_code={cam.camera_code}, auto_start={payload.auto_start}",
        actor=admin,
        ip_address=request.client.host if request.client else None,
    )
    return {"camera": _serialize_camera(cam, bridge)}


@router.post("/upload-video", status_code=status.HTTP_201_CREATED)
async def upload_camera_video(
    request: Request,
    file: UploadFile = File(...),
    camera_code: str = Form(...),
    name: str = Form(...),
    location: str = Form(""),
    auto_start: bool = Form(True),
    actor: User = Depends(require_roles("ADMIN", "OPERATOR")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Upload a local video file and register/start it as a camera stream."""
    code = camera_code.strip()
    if not re.match(r"^[a-zA-Z0-9_.-]+$", code):
        raise HTTPException(status_code=422, detail="Invalid camera_code format.")

    orig_name = Path(file.filename or "video.mp4").name
    ext = Path(orig_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTS)}",
        )

    if db.query(Camera).filter(Camera.camera_code == code).first() is not None:
        raise HTTPException(status_code=409, detail=f"Camera '{code}' already exists.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(orig_name).stem)[:48]
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}_{safe_stem}{ext}"

    total = 0
    with dest.open("wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                out_f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video exceeds 200 MB upload limit.")
            out_f.write(chunk)

    cam = Camera(
        camera_code=code,
        name=name.strip() or code,
        source=str(dest),
        location=location.strip(),
        enabled=True,
        status="ONLINE" if auto_start else "OFFLINE",
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    bridge = _bridge(request)
    try:
        bridge.register_or_sync_camera(cam, auto_start=auto_start)
    except Exception as exc:
        cam.status = "ERROR"
        db.commit()
        raise HTTPException(status_code=422, detail=f"Camera AI registration failed: {exc}") from exc

    record_audit_event(
        db,
        action="CAMERA_VIDEO_UPLOADED",
        resource_type="camera",
        resource_id=str(cam.id),
        details=f"camera_code={code}, file={dest.name}, bytes={total}",
        actor=actor,
        ip_address=request.client.host if request.client else None,
    )
    return {"camera": _serialize_camera(cam, bridge)}


@router.get("/{camera_id}")
def get_camera(
    camera_id: int,
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return {"camera": _serialize_camera(cam, _bridge(request))}


@router.put("/{camera_id}")
def update_camera(
    camera_id: int,
    payload: CameraUpdateRequest,
    request: Request,
    admin: User = Depends(require_roles("ADMIN")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found.")

    if payload.name is not None:
        cam.name = payload.name.strip()
    if payload.source is not None:
        cam.source = str(payload.source)
    if payload.location is not None:
        cam.location = payload.location.strip()
    if payload.fps is not None:
        cam.fps = payload.fps
    if payload.enabled is not None:
        cam.enabled = payload.enabled

    bridge = _bridge(request)
    if payload.action == "start":
        bridge.register_or_sync_camera(cam, auto_start=True)
        cam.status = "ONLINE"
    elif payload.action == "stop":
        try:
            bridge.stop_camera(cam.camera_code)
        except KeyError:
            pass
        cam.status = "OFFLINE"

    db.commit()
    db.refresh(cam)
    record_audit_event(
        db,
        action="CAMERA_UPDATED",
        resource_type="camera",
        resource_id=str(cam.id),
        details=f"camera_code={cam.camera_code}, status={cam.status}",
        actor=admin,
        ip_address=request.client.host if request.client else None,
    )
    return {"camera": _serialize_camera(cam, bridge)}


@router.delete("/{camera_id}")
def delete_camera(
    camera_id: int,
    request: Request,
    admin: User = Depends(require_roles("ADMIN")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found.")

    code = cam.camera_code
    _bridge(request).remove_camera(code)
    db.delete(cam)
    db.commit()
    record_audit_event(
        db,
        action="CAMERA_DELETED",
        resource_type="camera",
        resource_id=str(camera_id),
        details=f"camera_code={code}",
        actor=admin,
        ip_address=request.client.host if request.client else None,
    )
    return {"message": "Camera deleted."}


@router.get("/{camera_id}/frame")
def get_camera_frame(
    camera_id: int,
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> Response:
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    jpg = _bridge(request).state.latest_frames_jpeg.get(cam.camera_code)
    if jpg is None:
        raise HTTPException(status_code=404, detail="No live frame available for this camera.")
    return Response(content=jpg, media_type="image/jpeg")
