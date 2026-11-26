"""Camera management and live WebSocket streaming routes for ARGUS-X."""

from __future__ import annotations

import asyncio
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from api.state import ArgusServiceState

router = APIRouter(tags=["cameras"])


class CameraRegisterRequest(BaseModel):
    """Validated payload for registering a camera stream."""

    camera_id: str = Field(..., min_length=1, max_length=64)
    source: Optional[Union[str, int]] = None
    name: Optional[str] = Field(default=None, max_length=128)
    fps: Optional[float] = Field(default=None, gt=0.0)
    enabled: bool = True
    auto_start: bool = False


def _get_state(request: Request) -> ArgusServiceState:
    return request.app.state.argus


@router.get("/cameras")
def list_cameras(request: Request) -> dict[str, object]:
    """List all registered cameras, their status, and active detections."""
    state = _get_state(request)
    cameras = state.list_cameras()
    return {"cameras": cameras, "count": len(cameras)}


@router.post("/cameras/register", status_code=201)
def register_camera(payload: CameraRegisterRequest, request: Request) -> dict[str, object]:
    """Register a camera stream with the backend CameraManager."""
    state = _get_state(request)
    try:
        cam = state.register_camera(
            camera_id=payload.camera_id.strip(),
            source=payload.source,
            name=payload.name,
            fps=payload.fps,
            enabled=payload.enabled,
            auto_start=payload.auto_start,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"camera": cam}


@router.post("/cameras/{camera_id}/disconnect")
def disconnect_camera(camera_id: str, request: Request) -> dict[str, object]:
    """Gracefully disconnect a camera stream without deleting identity history."""
    state = _get_state(request)
    try:
        return state.disconnect_camera(camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.") from exc


@router.get("/cameras/{camera_id}/frame")
def get_camera_frame(camera_id: str, request: Request) -> Response:
    """Return the latest annotated JPEG frame for a registered camera."""
    state = _get_state(request)
    try:
        state.camera_manager.get_camera(camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.") from exc

    jpg = state.latest_frames_jpeg.get(camera_id)
    if jpg is None:
        raise HTTPException(status_code=404, detail=f"No frame available for '{camera_id}'.")
    return Response(content=jpg, media_type="image/jpeg")


@router.websocket("/ws/cameras/{camera_id}")
async def stream_camera_ws(websocket: WebSocket, camera_id: str) -> None:
    """Stream live annotated frames and detection overlays over WebSocket."""
    state: ArgusServiceState = websocket.app.state.argus
    try:
        state.camera_manager.get_camera(camera_id)
    except KeyError:
        await websocket.close(code=1008, reason=f"Camera '{camera_id}' not found.")
        return

    await websocket.accept()
    try:
        while True:
            payload = state.get_camera_stream_payload(camera_id)
            await websocket.send_json(payload)
            if not payload["is_running"]:
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
    except KeyError:
        await websocket.close(code=1001, reason="Camera unregistered.")
