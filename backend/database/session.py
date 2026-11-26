"""Database engine and session management for ARGUS-X."""

from __future__ import annotations

from collections.abc import Generator
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_backend_settings


def normalize_database_url(raw_url: str) -> str:
    """Normalize PostgreSQL/Neon connection strings for SQLAlchemy + psycopg2."""
    url = raw_url.strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://") :]

    if "neon.tech" in url and "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def build_engine(database_url: Optional[str] = None) -> Engine:
    """Create a SQLAlchemy engine for Neon/PostgreSQL or SQLite."""
    url = normalize_database_url(database_url or get_backend_settings().database_url)
    is_sqlite = url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    engine_kwargs = {"connect_args": connect_args, "pool_pre_ping": True, "future": True}
    if not is_sqlite:
        # Neon serverless Postgres closes idle connections after ~5 minutes
        engine_kwargs["pool_recycle"] = 300
        engine_kwargs["pool_size"] = 5
        engine_kwargs["max_overflow"] = 10

    engine = create_engine(url, **engine_kwargs)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


_default_engine = build_engine()
SessionLocal = sessionmaker(bind=_default_engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
