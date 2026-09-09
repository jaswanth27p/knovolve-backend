"""add session_started_at to refresh tokens

Revision ID: 2b9f8c1a5e42
Revises: 1a5d6e5d91ff
Create Date: 2026-09-08 21:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '2b9f8c1a5e42'
down_revision: Union[str, Sequence[str], None] = '1a5d6e5d91ff'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Existing rows predate the concept of a session birth; backfill them to
    # their own created_at (the closest thing to a chain start we have) so the
    # column is non-null, then add the NOT NULL constraint for new writes.
    op.add_column('refresh_tokens', sa.Column('session_started_at', sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE refresh_tokens SET session_started_at = created_at WHERE session_started_at IS NULL")
    op.alter_column('refresh_tokens', 'session_started_at', nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('refresh_tokens', 'session_started_at')