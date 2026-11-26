"""Main FastAPI application for the ARGUS-X Full-Stack Platform.

Mounts all new /api/* full-stack routes (Auth, Users, Cameras, Monitoring,
Identities, Analytics, Audit, Settings) alongside the existing api/ routes,
sharing a single ArgusServiceState AI engine instance.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.analytics.router import router as analytics_router
from backend.audit.router import router as audit_router
from backend.auth.router import router as auth_router
from backend.benchmarks.router import router as benchmarks_router
from backend.cameras.router import router as cameras_router
from backend.config import get_backend_settings
from backend.database.base import Base
from backend.database.session import SessionLocal, build_engine
from backend.datasets.router import router as datasets_router
from backend.identities.router import router as identities_router
from backend.monitoring.router import router as monitoring_router
from backend.multicamera.router import router as multicamera_router
from backend.services.seed import seed_database
from backend.users.router import router as users_router
from backend.videos.router import router as videos_router

# Detect if we are running in a resource-limited cloud environment (Render, etc.)
_CLOUD_MODE = os.getenv("RENDER", "").lower() in ("1", "true", "yes") or \
              os.getenv("ARGUS_CLOUD_MODE", "").lower() in ("1", "true", "yes")


def _build_ai_bridge(service_state=None):
    """Lazy-load heavy AI imports only when NOT in cloud/API-only mode."""
    if _CLOUD_MODE:
        # Return a lightweight stub so the API server starts without loading GPU models
        from datetime import datetime, timezone
        from backend.models import Camera, Identity, IdentityObservation, RecoveryEvent

        class _AIStub:
            def get_statistics(self):
                return {"mode": "cloud-api-only", "ai_enabled": False}

            def get_camera_stream_payload(self, camera_code: str):
                raise KeyError(camera_code)

            def close(self):
                pass

        class _AIBridgeStub:
            state = _AIStub()

            def register_or_sync_camera(self, *a, **kw):
                pass

            def get_monitoring_snapshot(self, db):
                cams = db.query(Camera).order_by(Camera.id.asc()).all()
                ids = db.query(Identity).order_by(Identity.id.asc()).all()
                uncertain_count = db.query(IdentityObservation).filter(
                    IdentityObservation.decision == "UNCERTAIN"
                ).count()
                return {
                    "status": "ok",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "statistics": {
                        "total_cameras": len(cams),
                        "online_cameras": sum(1 for c in cams if c.status == "ONLINE"),
                        "offline_cameras": sum(1 for c in cams if c.status != "ONLINE"),
                        "total_identities": len(ids),
                        "active_identities": sum(1 for i in ids if i.lifecycle_state == "ACTIVE"),
                        "temporarily_lost_identities": sum(1 for i in ids if i.lifecycle_state == "TEMPORARILY_LOST"),
                        "recovered_identities": sum(1 for i in ids if i.lifecycle_state == "RECOVERED"),
                        "expired_identities": sum(1 for i in ids if i.lifecycle_state == "EXPIRED"),
                        "uncertain_observations": uncertain_count,
                        "total_observations_in_memory": db.query(IdentityObservation).count(),
                        "total_conflicts": sum(1 for i in ids if i.has_conflict),
                    },
                    "cameras": [
                        {
                            "id": c.camera_code,
                            "name": c.name,
                            "enabled": c.enabled,
                            "online": c.status == "ONLINE",
                            "fps": c.fps,
                            "active_tracks": 0,
                        }
                        for c in cams
                    ],
                    "tracking_events": [],
                    "identities": [
                        {
                            "global_id": i.global_id,
                            "lifecycle_state": i.lifecycle_state,
                            "last_seen_camera": i.last_seen_camera,
                            "confidence": i.confidence,
                            "has_conflict": i.has_conflict,
                            "observation_count": i.observation_count,
                        }
                        for i in ids
                    ],
                    "uncertain_alerts": [],
                }

            def list_safe_identities(self, db):
                rows = db.query(Identity).order_by(Identity.updated_at.desc()).all()
                return [
                    {
                        "id": r.id,
                        "global_id": r.global_id,
                        "display_label": r.display_label or r.global_id,
                        "lifecycle_state": r.lifecycle_state,
                        "last_seen_camera": r.last_seen_camera,
                        "confidence": r.confidence,
                        "observation_count": r.observation_count,
                        "has_conflict": r.has_conflict,
                        "camera_history": r.camera_history or [],
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                    }
                    for r in rows
                ]

            def list_uncertain_queue(self, db):
                rows = (
                    db.query(IdentityObservation)
                    .filter(IdentityObservation.decision == "UNCERTAIN")
                    .order_by(IdentityObservation.id.desc())
                    .limit(100)
                    .all()
                )
                return [
                    {
                        "id": r.id,
                        "camera_id": r.camera_code,
                        "track_id": r.track_id,
                        "candidate_global_id": r.global_id,
                        "similarity_score": r.confidence,
                        "decision": r.decision,
                        "reason": r.reason,
                    }
                    for r in rows
                ]

            def close(self):
                pass

        return _AIBridgeStub()

    # Full AI stack – only imported on local / GPU servers
    from api.state import ArgusServiceState
    from backend.services.ai_engine import ArgusAIEngineBridge
    return ArgusAIEngineBridge(service_state=service_state)


def create_fullstack_app(
    service_state=None,
    init_db: bool = True,
) -> FastAPI:
    """Create the unified ARGUS-X FastAPI application."""
    settings = get_backend_settings()
    bridge = _build_ai_bridge(service_state)

    if init_db:
        engine = build_engine(settings.database_url)
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            seed_database(db, create_bootstrap_admin=True)
            from backend.models import BenchmarkJob, Camera, UploadedVideo
            from pathlib import Path
            # Cleanup abandoned/interrupted jobs from prior server runs
            db.query(BenchmarkJob).filter(BenchmarkJob.status.in_(["QUEUED", "RUNNING"])).update(
                {"status": "FAILED", "error_message": "Interrupted by server restart"},
                synchronize_session=False,
            )
            db.query(UploadedVideo).filter(UploadedVideo.analysis_status.in_(["QUEUED", "PROCESSING"])).update(
                {"analysis_status": "FAILED", "error_message": "Interrupted by server restart"},
                synchronize_session=False,
            )
            db.commit()
            if not _CLOUD_MODE:
                for cam in db.query(Camera).filter(Camera.enabled.is_(True)).all():
                    if cam.source and Path(cam.source).is_file():
                        bridge.register_or_sync_camera(cam, auto_start=True)
        finally:
            db.close()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        if service_state is None:
            bridge.close()

    application = FastAPI(
        title="ARGUS-X Full-Stack Platform API",
        description="Track the Untrackable — Multi-Camera Identity Recovery Platform",
        version="2.0.0",
        lifespan=lifespan,
    )
    application.state.argus = bridge.state
    application.state.ai_bridge = bridge

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_origin_regex=r"https://argus-[a-z0-9-]+-sanjaygm112005-uxs-projects\.vercel\.app|https://argus-x-rho\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health", tags=["health"])
    def api_health() -> dict[str, object]:
        stats = bridge.state.get_statistics()
        return {"status": "ok", "service": "ARGUS-X", "cloud_mode": _CLOUD_MODE, "ai_stats": stats}

    # Full-stack authenticated /api/* routes
    application.include_router(auth_router)
    application.include_router(users_router)
    application.include_router(cameras_router)
    application.include_router(monitoring_router)
    application.include_router(identities_router)
    application.include_router(analytics_router)
    application.include_router(audit_router)
    application.include_router(datasets_router)
    application.include_router(benchmarks_router)
    application.include_router(videos_router)
    application.include_router(multicamera_router)

    if not _CLOUD_MODE:
        # Legacy routes use full AI state – skip on cloud/API-only instances
        from api.routes.audit import router as legacy_audit_router
        from api.routes.cameras import router as legacy_cameras_router
        from api.routes.identities import router as legacy_identities_router
        application.include_router(legacy_cameras_router)
        application.include_router(legacy_identities_router)
        application.include_router(legacy_audit_router)

    return application


app = create_fullstack_app()
