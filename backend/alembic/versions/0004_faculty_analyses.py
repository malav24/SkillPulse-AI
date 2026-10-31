"""faculty analyses (simple upload → compare → recommend workflow)

Revision ID: 0004
Revises: 0003
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "faculty_analyses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("course_name", sa.String(length=300), nullable=False),
        sa.Column("institution", sa.String(length=300), nullable=True),
        sa.Column("job_field", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("step", sa.String(length=40), nullable=True),
        sa.Column("message", sa.String(length=500), nullable=True),
        sa.Column("gap_run_id", sa.Integer(), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('queued', 'processing', 'done', 'failed')", name="ck_faculty_status"),
        sa.CheckConstraint("job_field IN ('it', 'all')", name="ck_faculty_job_field"),
        sa.ForeignKeyConstraint(["document_id"], ["syllabus_documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["gap_run_id"], ["gap_analysis_runs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_faculty_analyses_document_id", "faculty_analyses", ["document_id"])


def downgrade() -> None:
    op.drop_index("ix_faculty_analyses_document_id", table_name="faculty_analyses")
    op.drop_table("faculty_analyses")
