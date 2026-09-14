"""lowercase existing user emails

Revision ID: 6117dd7d05c3
Revises: b98e46d4518f
Create Date: 2026-09-14 23:41:48.903295

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6117dd7d05c3'
down_revision: Union[str, Sequence[str], None] = 'b98e46d4518f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # New registrations/logins now normalize email to lowercase (see
    # app/schemas/auth.py), so any pre-existing mixed-case row would
    # otherwise become permanently unreachable by login. Two existing rows
    # can collide once lowercased (e.g. "a@x.com" and "A@x.com" both
    # registered before this fix) — resolve by keeping the older account
    # (earlier created_at, ties broken by lower id) and dropping the rest.
    # This is a plain DELETE, not a cascading merge: if the row being
    # dropped already owns other data (courses, attempts, enrollments,
    # etc.), the FK constraint (no ON DELETE CASCADE on any users.id FK in
    # this schema) rejects the delete and the whole migration aborts —
    # deleting someone's actual learning history is a call for a human to
    # make deliberately, not something this migration does silently.
    op.execute(
        """
        DELETE FROM users u
        USING (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY lower(email)
                       ORDER BY created_at ASC, id ASC
                   ) AS rn
            FROM users
        ) ranked
        WHERE u.id = ranked.id AND ranked.rn > 1
        """
    )
    op.execute("UPDATE users SET email = lower(email)")


def downgrade() -> None:
    """Downgrade schema."""
    # Lowercasing is lossy (original casing isn't recorded anywhere), so
    # there's nothing to restore.
    pass
