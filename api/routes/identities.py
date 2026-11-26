"""Global identity, unresolved observation, and manual review routes."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.state import ArgusServiceState

router = APIRouter(tags=["identities"])


class ManualReviewRequest(BaseModel):
    """Validated payload for operator manual review of an unresolved observation."""

    decision: Literal["ACCEPT", "REJECT"]
    target_global_id: Optional[str] = None
    reason: str = Field(..., min_length=3, max_length=256)
    operator_confirmed: bool = False


class ControlledMergeRequest(BaseModel):
    """Validated payload for operator-approved identity merge."""

    source_global_id: str = Field(..., min_length=1)
    target_global_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=3, max_length=256)
    operator_approved: bool = False


def _get_state(request: Request) -> ArgusServiceState:
    return request.app.state.argus


@router.get("/identities")
def list_identities(
    request: Request,
    state_filter: Optional[str] = Query(default=None, alias="state"),
    query: Optional[str] = Query(default=None),
) -> dict[str, object]:
    """List global identities with lifecycle state and per-camera track history."""
    service = _get_state(request)
    records = service.identity_manager.list_identities()
    results: list[dict[str, object]] = []
    for rec in records:
        if state_filter and rec.state.value != state_filter.upper():
            continue
        if query and query.lower() not in rec.global_id.lower():
            continue
        obs = service.identity_manager.memory.get_identity_observations(rec.global_id)
        results.append(
            {
                "global_id": rec.global_id,
                "state": rec.state.value,
                "created_at": rec.created_at,
                "last_seen_at": rec.last_seen_at,
                "current_camera_id": rec.current_camera_id,
                "camera_sequence": rec.camera_sequence,
                "camera_track_history": rec.camera_track_history,
                "observation_count": len(obs),
            }
        )
    return {"identities": results, "count": len(results)}


@router.get("/identities/{global_id}")
def get_identity_details(global_id: str, request: Request) -> dict[str, object]:
    """Return full lifecycle, observation history, and audit trail for a global ID."""
    service = _get_state(request)
    rec = service.identity_manager.get_identity(global_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Identity '{global_id}' not found.")

    obs = service.identity_manager.memory.get_identity_observations(global_id)
    audit_entries = service.identity_manager.get_audit_logs(global_id=global_id)
    return {
        "identity": {
            "global_id": rec.global_id,
            "state": rec.state.value,
            "created_at": rec.created_at,
            "last_seen_at": rec.last_seen_at,
            "current_camera_id": rec.current_camera_id,
            "camera_sequence": rec.camera_sequence,
            "camera_track_history": rec.camera_track_history,
        },
        "observations": [
            {
                "embedding_id": o.embedding_id,
                "vector_id": o.vector_id,
                "camera_id": o.camera_id,
                "local_track_id": o.local_track_id,
                "timestamp": o.timestamp,
            }
            for o in obs
        ],
        "audit_logs": [
            {
                "timestamp": a.timestamp,
                "camera_id": a.camera_id,
                "local_track_id": a.local_track_id,
                "global_id": a.global_id,
                "decision": a.decision,
                "appearance_score": a.appearance_score,
                "temporal_score": a.temporal_score,
                "camera_score": a.camera_score,
                "motion_score": a.motion_score,
                "combined_confidence": a.combined_confidence,
                "reason": a.reason,
            }
            for a in audit_entries
        ],
    }


@router.get("/unresolved")
def list_unresolved(
    request: Request,
    status: Optional[str] = Query(default="PENDING"),
) -> dict[str, object]:
    """List unresolved/uncertain observations awaiting evidence or manual review."""
    service = _get_state(request)
    status_norm = status.upper() if status and status.upper() != "ALL" else None
    items = service.identity_manager.list_unresolved_observations(status=status_norm)
    return {
        "unresolved": [
            {
                "observation_id": item.tracklet_uid,
                "tracklet_uid": item.tracklet_uid,
                "camera_id": item.camera_id,
                "local_track_id": item.local_track_id,
                "first_seen_at": item.first_seen_at,
                "last_seen_at": item.last_seen_at,
                "candidate_ids": item.candidate_ids,
                "confidence": item.confidence,
                "reason": item.reason,
                "status": item.status,
                "resolved_global_id": item.resolved_global_id,
            }
            for item in items
        ],
        "count": len(items),
    }


@router.post("/unresolved/{observation_id}/review")
def review_unresolved_observation(
    observation_id: str,
    payload: ManualReviewRequest,
    request: Request,
) -> dict[str, object]:
    """Protected manual review endpoint delegating strictly to GlobalIdentityManager."""
    if not payload.operator_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Manual review requires operator_confirmed=True.",
        )

    service = _get_state(request)
    try:
        updated = service.identity_manager.review_unresolved(
            tracklet_uid=observation_id,
            decision=payload.decision,
            target_global_id=payload.target_global_id,
            reason=payload.reason.strip(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation_id": updated.tracklet_uid,
        "status": updated.status,
        "resolved_global_id": updated.resolved_global_id,
        "reason": updated.reason,
    }


@router.post("/identities/merge")
def controlled_merge_identities(
    payload: ControlledMergeRequest,
    request: Request,
) -> dict[str, object]:
    """Protected controlled identity merge endpoint."""
    service = _get_state(request)
    try:
        service.identity_manager.merge_identities(
            source_global_id=payload.source_global_id,
            target_global_id=payload.target_global_id,
            reason=payload.reason.strip(),
            operator_approved=payload.operator_approved,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "merged_audited",
        "source_global_id": payload.source_global_id,
        "target_global_id": payload.target_global_id,
    }
