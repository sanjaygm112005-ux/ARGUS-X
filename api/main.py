"""FastAPI entrypoint for the ARGUS-X monitoring and identity-recovery backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request

from api.routes.audit import router as audit_router
from api.routes.cameras import router as cameras_router
from api.routes.identities import router as identities_router
from api.state import ArgusServiceState
from config import LOGS_DIR


def create_app(service_state: Optional[ArgusServiceState] = None) -> FastAPI:
    """Create and configure the ARGUS-X FastAPI application."""
    state = service_state or ArgusServiceState(
        db_path=LOGS_DIR / "argus_memory.db",
        index_path=LOGS_DIR / "argus_memory.faiss",
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        if service_state is None:
            application.state.argus.close()

    application = FastAPI(
        title="ARGUS-X API",
        description="Track the Untrackable — Multi-Camera Identity Recovery Backend",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.argus = state

    @application.get("/health", tags=["health"])
    def health_check(request: Request) -> dict[str, object]:
        stats = request.app.state.argus.get_statistics()
        return {
            "status": "ok",
            "service": "ARGUS-X",
            "total_cameras": stats["total_cameras"],
            "online_cameras": stats["online_cameras"],
            "total_identities": stats["total_identities"],
        }

    application.include_router(cameras_router)
    application.include_router(identities_router)
    application.include_router(audit_router)
    return application


app = create_app()
