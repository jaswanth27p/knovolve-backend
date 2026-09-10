"""add allow_duplicate to course_jobs

Revision ID: c4652b1e9d01
Revises: aa70ce221cc6
Create Date: 2026-09-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c4652b1e9d01'
down_revision: Union[str, Sequence[str], None] = 'aa70ce221cc6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'course_jobs',
        sa.Column('allow_duplicate', sa.Boolean(), nullable=False, server_default='false'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('course_jobs', 'allow_duplicate')