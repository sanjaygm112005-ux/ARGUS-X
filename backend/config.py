"""Environment-driven backend configuration for ARGUS-X."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class BackendSettings:
    """Immutable settings loaded from environment variables."""

    database_url: str
    ai_memory_db_path: Path
    ai_memory_index_path: Path
    jwt_secret_key: str
    jwt_algorithm: str
    access_token_expire_minutes: int
    cookie_secure: bool
    cookie_samesite: str
    login_rate_limit_attempts: int
    login_rate_limit_window_seconds: int
    cors_origins: tuple[str, ...]
    bootstrap_admin_email: str
    bootstrap_admin_username: str
    bootstrap_admin_password: str


@lru_cache(maxsize=1)
def get_backend_settings() -> BackendSettings:
    """Return cached backend settings from environment variables."""
    is_render = os.getenv("RENDER", "").lower() in ("1", "true", "yes")
    raw_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,https://argus-x-rho.vercel.app",
    )
    origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())

    return BackendSettings(
        database_url=os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'logs' / 'argus_app.db'}"),
        ai_memory_db_path=Path(
            os.getenv("AI_MEMORY_DB_PATH", str(PROJECT_ROOT / "logs" / "argus_memory.db"))
        ),
        ai_memory_index_path=Path(
            os.getenv("AI_MEMORY_INDEX_PATH", str(PROJECT_ROOT / "logs" / "argus_memory.faiss"))
        ),
        jwt_secret_key=os.getenv(
            "JWT_SECRET_KEY",
            "dev-only-argusx-secret-key-change-in-production-64-bytes-minimum",
        ),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")),
        cookie_secure=os.getenv("COOKIE_SECURE", "true" if is_render else "false").lower() in ("1", "true", "yes"),
        cookie_samesite=os.getenv("COOKIE_SAMESITE", "none" if is_render else "lax").lower(),
        login_rate_limit_attempts=int(os.getenv("LOGIN_RATE_LIMIT_ATTEMPTS", "5")),
        login_rate_limit_window_seconds=int(os.getenv("LOGIN_RATE_LIMIT_WINDOW_SECONDS", "300")),
        cors_origins=origins,
        bootstrap_admin_email=os.getenv("BOOTSTRAP_ADMIN_EMAIL", "admin@argusx.local"),
        bootstrap_admin_username=os.getenv("BOOTSTRAP_ADMIN_USERNAME", "admin"),
        bootstrap_admin_password=os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "ChangeMeNow!123"),
    )
