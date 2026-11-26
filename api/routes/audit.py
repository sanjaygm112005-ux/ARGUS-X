"""Audit log and system statistics routes for ARGUS-X."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request

from api.state import ArgusServiceState

router = APIRouter(tags=["audit"])


def _get_state(request: Request) -> ArgusServiceState:
    return request.app.state.argus


@router.get("/audit")
def get_audit_logs(
    request: Request,
    global_id: Optional[str] = Query(default=None),
    camera_id: Optional[str] = Query(default=None),
    decision: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, object]:
    """Return persisted audit log entries with confidence component breakdowns."""
    service = _get_state(request)
    entries = service.identity_manager.get_audit_logs(
        global_id=global_id,
        camera_id=camera_id,
        decision=decision,
    )
    sliced = entries[-limit:]
    return {
        "audit_logs": [
            {
                "timestamp": e.timestamp,
                "camera_id": e.camera_id,
                "local_track_id": e.local_track_id,
                "global_id": e.global_id,
                "decision": e.decision,
                "appearance_score": e.appearance_score,
                "temporal_score": e.temporal_score,
                "camera_score": e.camera_score,
                "motion_score": e.motion_score,
                "combined_confidence": e.combined_confidence,
                "reason": e.reason,
            }
            for e in reversed(sliced)
        ],
        "count": len(sliced),
    }


@router.get("/statistics")
def get_statistics(request: Request) -> dict[str, object]:
    """Return aggregated overview KPIs for the monitoring dashboard."""
    service = _get_state(request)
    return service.get_statistics()
