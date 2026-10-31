"""index job_postings.duplicate_cluster_id (dedup speed on large datasets)

Revision ID: 0003
Revises: 0002
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_job_postings_duplicate_cluster_id", "job_postings", ["duplicate_cluster_id"])


def downgrade() -> None:
    op.drop_index("ix_job_postings_duplicate_cluster_id", table_name="job_postings")
