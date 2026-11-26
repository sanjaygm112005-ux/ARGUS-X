"""Initial ARGUS-X application schema.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-25
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=32), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_roles_name", "roles", ["name"])

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("full_name", sa.String(length=128), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("role_id", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("reset_token_hash", sa.String(length=128), nullable=True),
        sa.Column("reset_token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("username"),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_role_id", "users", ["role_id"])

    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("session_jti", sa.String(length=128), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("remember_me", sa.Boolean(), nullable=False),
        sa.Column("is_revoked", sa.Boolean(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_jti"),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])
    op.create_index("ix_user_sessions_session_jti", "user_sessions", ["session_jti"])

    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("camera_code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("source", sa.String(length=512), nullable=True),
        sa.Column("location", sa.String(length=128), nullable=False),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("camera_code"),
    )
    op.create_index("ix_cameras_camera_code", "cameras", ["camera_code"])
    op.create_index("ix_cameras_status", "cameras", ["status"])

    op.create_table(
        "identities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("global_id", sa.String(length=64), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        sa.Column("current_camera_code", sa.String(length=64), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("has_conflict", sa.Boolean(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("global_id"),
    )
    op.create_index("ix_identities_global_id", "identities", ["global_id"])
    op.create_index("ix_identities_lifecycle_state", "identities", ["lifecycle_state"])
    op.create_index("ix_identities_current_camera_code", "identities", ["current_camera_code"])

    op.create_table(
        "tracking_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("camera_id", sa.Integer(), nullable=False),
        sa.Column("local_track_id", sa.Integer(), nullable=False),
        sa.Column("global_id", sa.String(length=64), nullable=True),
        sa.Column("bbox_x1", sa.Float(), nullable=False),
        sa.Column("bbox_y1", sa.Float(), nullable=False),
        sa.Column("bbox_x2", sa.Float(), nullable=False),
        sa.Column("bbox_y2", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tracking_events_camera_id", "tracking_events", ["camera_id"])
    op.create_index("ix_tracking_events_global_id", "tracking_events", ["global_id"])
    op.create_index("ix_tracking_events_cam_ts", "tracking_events", ["camera_id", "observed_at"])

    op.create_table(
        "identity_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("identity_id", sa.Integer(), nullable=False),
        sa.Column("camera_code", sa.String(length=64), nullable=False),
        sa.Column("local_track_id", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["identity_id"], ["identities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_identity_observations_identity_id", "identity_observations", ["identity_id"])
    op.create_index("ix_identity_observations_camera_code", "identity_observations", ["camera_code"])
    op.create_index("ix_identity_obs_identity_ts", "identity_observations", ["identity_id", "observed_at"])

    op.create_table(
        "recovery_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("identity_id", sa.Integer(), nullable=True),
        sa.Column("global_id", sa.String(length=64), nullable=True),
        sa.Column("from_camera_code", sa.String(length=64), nullable=True),
        sa.Column("to_camera_code", sa.String(length=64), nullable=False),
        sa.Column("local_track_id", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("reviewed_by_user_id", sa.Integer(), nullable=True),
        sa.Column("reason", sa.String(length=512), nullable=False),
        sa.Column("recovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["identity_id"], ["identities.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["reviewed_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recovery_events_identity_id", "recovery_events", ["identity_id"])
    op.create_index("ix_recovery_events_global_id", "recovery_events", ["global_id"])
    op.create_index("ix_recovery_events_decision", "recovery_events", ["decision"])
    op.create_index("ix_recovery_events_review_status", "recovery_events", ["review_status"])
    op.create_index("ix_recovery_events_ts", "recovery_events", ["recovered_at"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("actor_username", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("resource_type", sa.String(length=64), nullable=False),
        sa.Column("resource_id", sa.String(length=128), nullable=False),
        sa.Column("details", sa.Text(), nullable=False),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

    op.create_table(
        "system_settings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", sa.String(length=512), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key", name="uq_system_settings_key"),
    )
    op.create_index("ix_system_settings_key", "system_settings", ["key"])


def downgrade() -> None:
    op.drop_table("system_settings")
    op.drop_table("audit_logs")
    op.drop_table("recovery_events")
    op.drop_table("identity_observations")
    op.drop_table("tracking_events")
    op.drop_table("identities")
    op.drop_table("cameras")
    op.drop_table("user_sessions")
    op.drop_table("users")
    op.drop_table("roles")
