"""add chain_id to refresh tokens

Revision ID: 3c0d91d7f6a4
Revises: 2b9f8c1a5e42
Create Date: 2026-09-08 22:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '3c0d91d7f6a4'
down_revision: Union[str, Sequence[str], None] = '2b9f8c1a5e42'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Existing rows predate chain identity. Backfill each existing chain with a
    # per-chain UUID so chain-aware revocation doesn't accidentally strand any
    # legacy session: group by (user_id, session_started_at) — every token in a
    # rotation chain shares both — and give each group one chain_id.
    op.add_column('refresh_tokens', sa.Column('chain_id', sa.String(length=36), nullable=True))
    op.execute(
        """
        UPDATE refresh_tokens SET chain_id = sub.chain_id FROM (
            SELECT DISTINCT user_id, session_started_at,
                   uuid_in(md5(random()::text || clock_timestamp()::text)::cstring)::text AS chain_id
            FROM refresh_tokens
        ) AS sub
        WHERE refresh_tokens.user_id = sub.user_id
          AND refresh_tokens.session_started_at = sub.session_started_at
        """
    )
    op.alter_column('refresh_tokens', 'chain_id', nullable=False)
    op.create_index('ix_refresh_tokens_chain_id', 'refresh_tokens', ['chain_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_refresh_tokens_chain_id', table_name='refresh_tokens')
    op.drop_column('refresh_tokens', 'chain_id')