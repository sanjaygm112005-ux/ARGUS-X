"""Analytics routes (/api/analytics) grounded in real DB + AI engine state."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from backend.auth.security import require_roles
from backend.config import PROJECT_ROOT
from backend.database.session import get_db
from backend.models import BenchmarkResult, Camera, Identity, RecoveryEvent, TrackingEvent, User
from backend.services.ai_engine import ArgusAIEngineBridge

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _bridge(request: Request) -> ArgusAIEngineBridge:
    return request.app.state.ai_bridge


@router.get("/overview")
def analytics_overview(
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    bridge = _bridge(request)
    snapshot = bridge.get_monitoring_snapshot(db)
    total_tracking_events = db.query(TrackingEvent).count()
    total_recovery_events = db.query(RecoveryEvent).count()
    conflict_count = db.query(Identity).filter(Identity.has_conflict.is_(True)).count()

    cameras = snapshot["cameras"]
    camera_distribution = [
        {"camera": c["camera_code"], "active_tracks": c["active_tracks"], "status": c["status"]}
        for c in cameras
    ]

    return {
        "statistics": snapshot["statistics"],
        "total_tracking_events": total_tracking_events,
        "total_recovery_events": total_recovery_events,
        "conflict_count": conflict_count,
        "camera_distribution": camera_distribution,
    }


@router.get("/performance")
def analytics_performance(
    request: Request,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Return measured benchmark results if present on disk, never fabricated."""
    real_json = PROJECT_ROOT / "evaluation" / "results" / "real_video_validation.json"
    real_report = None
    if real_json.is_file():
        try:
            real_report = json.loads(real_json.read_text(encoding="utf-8"))
        except Exception:
            real_report = None

    bench_rows = db.query(BenchmarkResult).order_by(BenchmarkResult.created_at.desc()).limit(25).all()
    mot17_benchmarks = [
        {
            "id": b.id,
            "job_id": b.job_id,
            "sequence_name": b.sequence_name,
            "mota": b.mota,
            "idf1": b.idf1,
            "hota": b.hota,
            "id_switches": b.id_switches,
            "false_positives": b.false_positives,
            "false_negatives": b.false_negatives,
            "fps": b.fps,
            "processing_latency_ms": b.processing_latency_ms,
            "frames_evaluated": b.frames_evaluated,
            "created_at": b.created_at.isoformat(),
        }
        for b in bench_rows
    ]

    return {
        "real_video_validation": real_report,
        "real_world_status": "COMPLETED" if real_report else "PENDING_ANNOTATED_EVALUATION",
        "mot17_benchmarks": mot17_benchmarks,
        "registered_cameras": db.query(Camera).count(),
        "live_stats": _bridge(request).state.get_statistics(),
    }
