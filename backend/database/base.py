"""SQLAlchemy declarative base for ARGUS-X application database."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all application ORM models."""
