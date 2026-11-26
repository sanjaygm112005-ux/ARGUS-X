"""Add datasets, benchmark_jobs, benchmark_results, and uploaded_videos tables.

Revision ID: 0002_platform_upgrade
Revises: 0001_initial_schema
Create Date: 2026-11-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_platform_upgrade"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("dataset_type", sa.String(length=32), nullable=False),
        sa.Column("root_path", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("sequence_count", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("last_scanned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_datasets_name", "datasets", ["name"])
    op.create_index("ix_datasets_status", "datasets", ["status"])

    op.create_table(
        "benchmark_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_uid", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=True),
        sa.Column("dataset_name", sa.String(length=64), nullable=False),
        sa.Column("sequence_name", sa.String(length=64), nullable=False),
        sa.Column("detector_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("processed_frames", sa.Integer(), nullable=False),
        sa.Column("total_frames", sa.Integer(), nullable=False),
        sa.Column("max_frames", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_uid"),
    )
    op.create_index("ix_benchmark_jobs_job_uid", "benchmark_jobs", ["job_uid"])
    op.create_index("ix_benchmark_jobs_dataset_id", "benchmark_jobs", ["dataset_id"])
    op.create_index("ix_benchmark_jobs_dataset_name", "benchmark_jobs", ["dataset_name"])
    op.create_index("ix_benchmark_jobs_sequence_name", "benchmark_jobs", ["sequence_name"])
    op.create_index("ix_benchmark_jobs_status", "benchmark_jobs", ["status"])
    op.create_index("ix_benchmark_jobs_created_by_user_id", "benchmark_jobs", ["created_by_user_id"])
    op.create_index("ix_benchmark_jobs_status_created", "benchmark_jobs", ["status", "created_at"])

    op.create_table(
        "benchmark_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("sequence_name", sa.String(length=64), nullable=False),
        sa.Column("mota", sa.Float(), nullable=False),
        sa.Column("idf1", sa.Float(), nullable=False),
        sa.Column("hota", sa.Float(), nullable=False),
        sa.Column("id_switches", sa.Integer(), nullable=False),
        sa.Column("false_positives", sa.Integer(), nullable=False),
        sa.Column("false_negatives", sa.Integer(), nullable=False),
        sa.Column("true_positives", sa.Integer(), nullable=False),
        sa.Column("total_gt", sa.Integer(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("processing_latency_ms", sa.Float(), nullable=False),
        sa.Column("frames_evaluated", sa.Integer(), nullable=False),
        sa.Column("details_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["benchmark_jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_benchmark_results_job_id", "benchmark_results", ["job_id"])
    op.create_index("ix_benchmark_results_sequence_name", "benchmark_results", ["sequence_name"])

    op.create_table(
        "uploaded_videos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("video_uid", sa.String(length=64), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("processed_path", sa.String(length=512), nullable=True),
        sa.Column("camera_label", sa.String(length=128), nullable=False),
        sa.Column("camera_code", sa.String(length=64), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("total_frames", sa.Integer(), nullable=False),
        sa.Column("analysis_status", sa.String(length=32), nullable=False),
        sa.Column("analysis_progress", sa.Float(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("results_json", sa.Text(), nullable=True),
        sa.Column("uploaded_by_user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["uploaded_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("video_uid"),
    )
    op.create_index("ix_uploaded_videos_video_uid", "uploaded_videos", ["video_uid"])
    op.create_index("ix_uploaded_videos_camera_label", "uploaded_videos", ["camera_label"])
    op.create_index("ix_uploaded_videos_camera_code", "uploaded_videos", ["camera_code"])
    op.create_index("ix_uploaded_videos_analysis_status", "uploaded_videos", ["analysis_status"])
    op.create_index("ix_uploaded_videos_uploaded_by_user_id", "uploaded_videos", ["uploaded_by_user_id"])
    op.create_index("ix_uploaded_videos_status_created", "uploaded_videos", ["analysis_status", "created_at"])


def downgrade() -> None:
    op.drop_table("uploaded_videos")
    op.drop_table("benchmark_results")
    op.drop_table("benchmark_jobs")
    op.drop_table("datasets")
