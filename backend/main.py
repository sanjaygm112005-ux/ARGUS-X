"""Main FastAPI application for the ARGUS-X Full-Stack Platform.

Mounts all new /api/* full-stack routes (Auth, Users, Cameras, Monitoring,
Identities, Analytics, Audit, Settings) alongside the existing api/ routes,
sharing a single ArgusServiceState AI engine instance.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.audit import router as legacy_audit_router
from api.routes.cameras import router as legacy_cameras_router
from api.routes.identities import router as legacy_identities_router
from api.state import ArgusServiceState
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
from backend.services.ai_engine import ArgusAIEngineBridge
from backend.services.seed import seed_database
from backend.users.router import router as users_router
from backend.videos.router import router as videos_router


def create_fullstack_app(
    service_state: Optional[ArgusServiceState] = None,
    init_db: bool = True,
) -> FastAPI:
    """Create the unified ARGUS-X FastAPI application."""
    settings = get_backend_settings()
    bridge = ArgusAIEngineBridge(service_state=service_state)

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
            for cam in db.query(Camera).filter(Camera.enabled.is_(True)).all():
                if cam.source and Path(cam.source).is_file():
                    bridge.register_or_sync_camera(cam, auto_start=True)
        finally:
            db.close()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        if service_state is None:
            application.state.ai_bridge.close()

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
        return {"status": "ok", "service": "ARGUS-X", "ai_stats": stats}

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

    # Preserve legacy /cameras, /identities, /audit endpoints for existing Streamlit dashboard
    application.include_router(legacy_cameras_router)
    application.include_router(legacy_identities_router)
    application.include_router(legacy_audit_router)

    return application


app = create_fullstack_app()
