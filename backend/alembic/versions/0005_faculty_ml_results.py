"""faculty analyses: any job role as target, entry-level filter, stored ML result

Revision ID: 0005
Revises: 0004
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("faculty_analyses") as b:
        b.drop_constraint("ck_faculty_job_field", type_="check")
        b.alter_column("job_field", existing_type=sa.String(length=10), type_=sa.String(length=60),
                       existing_nullable=False)
        b.add_column(sa.Column("entry_level", sa.Boolean(), nullable=False, server_default=sa.false()))
        b.add_column(sa.Column("result", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("faculty_analyses") as b:
        b.drop_column("result")
        b.drop_column("entry_level")
        b.alter_column("job_field", existing_type=sa.String(length=60), type_=sa.String(length=10),
                       existing_nullable=False)
        b.create_check_constraint("ck_faculty_job_field", "job_field IN ('it', 'all')")
