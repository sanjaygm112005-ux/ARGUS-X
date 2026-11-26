"""Identity recovery, uncertain queue, and manual review routes (/api/identities)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from backend.auth.security import require_roles, verify_csrf
from backend.database.session import get_db
from backend.models import Identity, RecoveryEvent, User
from backend.schemas import IdentityReviewPayload
from backend.services.audit_service import record_audit_event

if TYPE_CHECKING:
    from backend.services.ai_engine import ArgusAIEngineBridge

router = APIRouter(prefix="/api/identities", tags=["identities"])


def _bridge(request: Request) -> ArgusAIEngineBridge:
    return request.app.state.ai_bridge


@router.get("")
def list_identities(
    request: Request,
    state_filter: Optional[str] = Query(default=None, alias="state"),
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items = _bridge(request).list_safe_identities(db)
    if state_filter:
        sf = state_filter.strip().upper()
        items = [i for i in items if str(i["lifecycle_state"]).upper() == sf]
    return {"identities": items, "count": len(items)}


@router.get("/recovery-history")
def get_recovery_history(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    events: list[dict[str, object]] = []

    db_rows = (
        db.query(RecoveryEvent)
        .order_by(RecoveryEvent.recovered_at.desc())
        .limit(limit)
        .all()
    )
    for r in db_rows:
        events.append(
            {
                "id": f"db-{r.id}",
                "global_id": r.global_id,
                "from_camera_code": r.from_camera_code,
                "to_camera_code": r.to_camera_code,
                "local_track_id": r.local_track_id,
                "decision": r.decision,
                "confidence": round(float(r.confidence), 4),
                "review_status": r.review_status,
                "reason": r.reason,
                "timestamp": r.recovered_at.isoformat(),
            }
        )

    ai_logs = _bridge(request).state.identity_manager.get_audit_logs()
    for a in reversed(ai_logs[-limit:]):
        events.append(
            {
                "id": f"ai-{a.camera_id}-{a.local_track_id}-{a.timestamp}",
                "global_id": a.global_id,
                "from_camera_code": None,
                "to_camera_code": a.camera_id,
                "local_track_id": a.local_track_id,
                "decision": a.decision,
                "confidence": round(float(a.combined_confidence), 4),
                "review_status": "AUTO",
                "reason": a.reason,
                "timestamp": a.timestamp,
            }
        )

    return {"recovery_events": events[:limit], "count": len(events[:limit])}


@router.get("/uncertain")
def list_uncertain_identities(
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items = _bridge(request).list_uncertain_items(db)
    return {"uncertain": items, "count": len(items)}


@router.get("/{identity_id}")
def get_identity_detail(
    identity_id: str,
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    bridge = _bridge(request)
    with bridge.state._lock:
        rec = bridge.state.identity_manager.get_identity(identity_id)
        if rec is not None:
            obs = bridge.state.identity_manager.memory.get_identity_observations(identity_id)
            ai_logs = bridge.state.identity_manager.get_audit_logs(global_id=identity_id)
            return {
                "identity": {
                    "global_id": rec.global_id,
                    "lifecycle_state": rec.state.value,
                    "current_camera_code": rec.current_camera_id,
                    "camera_sequence": list(rec.camera_sequence),
                    "observation_count": len(obs),
                    "first_seen_at": rec.created_at,
                    "last_seen_at": rec.last_seen_at,
                },
                "observations": [
                    {
                        "camera_code": o.camera_id,
                        "local_track_id": o.local_track_id,
                        "embedding_id": o.embedding_id,
                        "timestamp": o.timestamp,
                    }
                    for o in obs
                ],
                "audit_trail": [
                    {
                        "timestamp": a.timestamp,
                        "camera_id": a.camera_id,
                        "local_track_id": a.local_track_id,
                        "decision": a.decision,
                        "confidence": round(float(a.combined_confidence), 4) if a.combined_confidence is not None else 1.0,
                        "reason": a.reason,
                    }
                    for a in ai_logs
                ],
            }

    row = db.query(Identity).filter(Identity.global_id == identity_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Identity '{identity_id}' not found.")

    return {
        "identity": {
            "global_id": row.global_id,
            "lifecycle_state": row.lifecycle_state,
            "current_camera_code": row.current_camera_code,
            "camera_sequence": [row.current_camera_code] if row.current_camera_code else [],
            "observation_count": len(row.observations),
            "has_conflict": row.has_conflict,
            "first_seen_at": row.first_seen_at.isoformat(),
            "last_seen_at": row.last_seen_at.isoformat(),
        },
        "observations": [
            {
                "camera_code": o.camera_code,
                "local_track_id": o.local_track_id,
                "confidence": o.confidence,
                "status": o.status,
                "timestamp": o.observed_at.isoformat(),
            }
            for o in row.observations
        ],
    }


@router.post("/{observation_id}/review")
def review_uncertain_identity(
    observation_id: str,
    payload: IdentityReviewPayload,
    request: Request,
    reviewer: User = Depends(require_roles("ADMIN", "OPERATOR")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Human-in-the-loop review of an uncertain identity (never auto-merges)."""
    if not payload.operator_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Explicit operator_confirmed=true is required for manual identity review.",
        )

    if observation_id.startswith("db-"):
        try:
            ev_id = int(observation_id.split("db-", 1)[1])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid observation ID.") from exc
        ev = db.query(RecoveryEvent).filter(RecoveryEvent.id == ev_id).first()
        if ev is None:
            raise HTTPException(status_code=404, detail="Uncertain event not found.")
        ev.review_status = "ACCEPTED" if payload.decision == "ACCEPT" else "REJECTED"
        ev.reviewed_by_user_id = reviewer.id
        if payload.decision == "ACCEPT" and payload.target_global_id:
            ev.global_id = payload.target_global_id
        ev.reason = f"{ev.reason} | Review ({reviewer.username}): {payload.reason}".strip(" |")
        db.commit()
        record_audit_event(
            db,
            action="IDENTITY_REVIEWED",
            resource_type="recovery_event",
            resource_id=observation_id,
            details=f"decision={payload.decision}, target={payload.target_global_id}",
            actor=reviewer,
            ip_address=request.client.host if request.client else None,
        )
        return {"observation_id": observation_id, "review_status": ev.review_status}

    bridge = _bridge(request)
    try:
        resolved = bridge.state.identity_manager.review_unresolved(
            tracklet_uid=observation_id,
            decision=payload.decision,
            target_global_id=payload.target_global_id,
            reason=payload.reason.strip(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Observation '{observation_id}' not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record_audit_event(
        db,
        action="IDENTITY_REVIEWED",
        resource_type="ai_observation",
        resource_id=observation_id,
        details=f"decision={payload.decision}, target={payload.target_global_id}",
        actor=reviewer,
        ip_address=request.client.host if request.client else None,
    )
    return {"observation_id": resolved.tracklet_uid, "review_status": resolved.status}
