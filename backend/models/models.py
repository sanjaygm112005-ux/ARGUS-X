"""SQLAlchemy ORM models for the ARGUS-X application database.

Note: User/auth and application telemetry tables live here (PostgreSQL/SQLite),
strictly separate from the AI engine's FAISS index and biometric SQLite store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    users: Mapped[list["User"]] = relationship("User", back_populates="role")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id", ondelete="RESTRICT"), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    reset_token_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    reset_token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    role: Mapped["Role"] = relationship("Role", back_populates="users")
    sessions: Mapped[list["UserSession"]] = relationship(
        "UserSession", back_populates="user", cascade="all, delete-orphan"
    )


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_jti: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    remember_me: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="sessions")


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    location: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="OFFLINE", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    tracking_events: Mapped[list["TrackingEvent"]] = relationship(
        "TrackingEvent", back_populates="camera", cascade="all, delete-orphan"
    )


class Identity(Base):
    __tablename__ = "identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    lifecycle_state: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE", index=True)
    current_camera_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    has_conflict: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    observations: Mapped[list["IdentityObservation"]] = relationship(
        "IdentityObservation", back_populates="identity", cascade="all, delete-orphan"
    )
    recovery_events: Mapped[list["RecoveryEvent"]] = relationship(
        "RecoveryEvent", back_populates="identity", cascade="all, delete-orphan"
    )


class TrackingEvent(Base):
    __tablename__ = "tracking_events"
    __table_args__ = (
        Index("ix_tracking_events_cam_ts", "camera_id", "observed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    local_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    global_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    decision: Mapped[str] = mapped_column(String(32), nullable=False, default="TRACKED")
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="tracking_events")


class IdentityObservation(Base):
    __tablename__ = "identity_observations"
    __table_args__ = (
        Index("ix_identity_obs_identity_ts", "identity_id", "observed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int] = mapped_column(ForeignKey("identities.id", ondelete="CASCADE"), nullable=False, index=True)
    camera_code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    local_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="CONFIRMED")
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    identity: Mapped["Identity"] = relationship("Identity", back_populates="observations")


class RecoveryEvent(Base):
    __tablename__ = "recovery_events"
    __table_args__ = (
        Index("ix_recovery_events_ts", "recovered_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("identities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    global_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    from_camera_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    to_camera_code: Mapped[str] = mapped_column(String(64), nullable=False)
    local_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, default="AUTO", index=True)
    reviewed_by_user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    reason: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    recovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    identity: Mapped[Optional["Identity"]] = relationship("Identity", back_populates="recovery_events")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_username: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    details: Mapped[str] = mapped_column(Text, nullable=False, default="")
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SystemSetting(Base):
    __tablename__ = "system_settings"
    __table_args__ = (UniqueConstraint("key", name="uq_system_settings_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    value: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    dataset_type: Mapped[str] = mapped_column(String(32), nullable=False, default="MOT17")
    root_path: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="AVAILABLE", index=True)
    sequence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    last_scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    benchmark_jobs: Mapped[list["BenchmarkJob"]] = relationship("BenchmarkJob", back_populates="dataset")


class BenchmarkJob(Base):
    __tablename__ = "benchmark_jobs"
    __table_args__ = (
        Index("ix_benchmark_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True
    )
    dataset_name: Mapped[str] = mapped_column(String(64), nullable=False, default="MOT17", index=True)
    sequence_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    detector_type: Mapped[str] = mapped_column(String(32), nullable=False, default="FRCNN")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED", index=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    processed_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    dataset: Mapped[Optional["Dataset"]] = relationship("Dataset", back_populates="benchmark_jobs")
    result: Mapped[Optional["BenchmarkResult"]] = relationship(
        "BenchmarkResult", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("benchmark_jobs.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    sequence_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    mota: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    idf1: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hota: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    id_switches: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    false_positives: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    false_negatives: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    true_positives: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_gt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    processing_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frames_evaluated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    job: Mapped["BenchmarkJob"] = relationship("BenchmarkJob", back_populates="result")


class UploadedVideo(Base):
    __tablename__ = "uploaded_videos"
    __table_args__ = (
        Index("ix_uploaded_videos_status_created", "analysis_status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    processed_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    camera_label: Mapped[str] = mapped_column(String(128), nullable=False, default="", index=True)
    camera_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    analysis_status: Mapped[str] = mapped_column(String(32), nullable=False, default="UPLOADED", index=True)
    analysis_progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    results_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    uploaded_by_user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
