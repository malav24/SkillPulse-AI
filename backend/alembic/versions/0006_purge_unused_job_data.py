"""remove job-market rows imported by earlier versions (now replaced by the trained-model artifacts)

The app no longer reads these tables: job postings are pre-processed into backend/artifacts/ by
`python -m ml.train_all`. Clearing them shrinks old databases from hundreds of MB to a few MB, which makes
every query fast again. The source CSVs in data/market/ are untouched.

Revision ID: 0006
Revises: 0005
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ["recommendations", "gap_results", "skill_mappings", "saved_analysis_configs", "job_skill_evidence",
           "job_postings", "job_duplicate_clusters", "job_import_batches", "job_datasets"]


def upgrade() -> None:
    op.execute("UPDATE faculty_analyses SET gap_run_id = NULL")
    for t in _TABLES:
        op.execute(f"DELETE FROM {t}")
    op.execute("DELETE FROM gap_analysis_runs")
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.get_context().autocommit_block():
            op.execute("VACUUM")


def downgrade() -> None:
    pass  # deleted rows were derived data; re-import is not needed by any version that uses the artifacts
