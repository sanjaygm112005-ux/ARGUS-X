"""Live monitoring endpoints and WebSocket feed (/api/monitoring, /ws/monitoring)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from backend.auth.security import ACCESS_COOKIE_NAME, decode_access_token, require_roles
from backend.database.session import SessionLocal, get_db
from backend.models import User, UserSession
from backend.services.ai_engine import ArgusAIEngineBridge

router = APIRouter(tags=["monitoring"])


def _bridge(request: Request) -> ArgusAIEngineBridge:
    return request.app.state.ai_bridge


@router.get("/api/monitoring/status")
def monitoring_status(
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    snapshot = _bridge(request).get_monitoring_snapshot(db)
    return {
        "status": snapshot["status"],
        "statistics": snapshot["statistics"],
        "timestamp": snapshot["timestamp"],
    }


@router.get("/api/monitoring/cameras")
def monitoring_cameras(
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    snapshot = _bridge(request).get_monitoring_snapshot(db)
    return {
        "cameras": snapshot["cameras"],
        "tracking_events": _bridge(request).get_recent_tracking_events(db, limit=30),
        "timestamp": snapshot["timestamp"],
    }


@router.get("/api/monitoring/identities")
def monitoring_identities(
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    snapshot = _bridge(request).get_monitoring_snapshot(db)
    return {
        "identities": snapshot["identities"],
        "uncertain_alerts": snapshot["uncertain_alerts"],
        "timestamp": snapshot["timestamp"],
    }


@router.websocket("/ws/monitoring")
async def monitoring_ws(websocket: WebSocket) -> None:
    """Live WebSocket feed pushing camera, identity, and alert updates."""
    token = websocket.cookies.get(ACCESS_COOKIE_NAME) or websocket.query_params.get("token")
    if token:
        try:
            payload = decode_access_token(token)
            user_id = int(str(payload["sub"]))
            jti = str(payload["jti"])
            role_name = str(payload["role"])
            if role_name in ("ADMIN", "OPERATOR"):
                db = SessionLocal()
                try:
                    sess = (
                        db.query(UserSession)
                        .filter(
                            UserSession.session_jti == jti,
                            UserSession.user_id == user_id,
                            UserSession.is_revoked.is_(False),
                        )
                        .first()
                    )
                finally:
                    db.close()
        except Exception:
            pass

    await websocket.accept()
    bridge: ArgusAIEngineBridge = websocket.app.state.ai_bridge
    try:
        while True:
            db_tick = SessionLocal()
            try:
                snapshot = bridge.get_monitoring_snapshot(db_tick)
            finally:
                db_tick.close()
            await websocket.send_json(snapshot)
            has_running = any(c.get("is_running") for c in snapshot.get("cameras", []))
            await asyncio.sleep(0.25 if has_running else 1.0)
    except WebSocketDisconnect:
        return
