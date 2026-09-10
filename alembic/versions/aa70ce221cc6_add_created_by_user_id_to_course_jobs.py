"""add created_by_user_id to course_jobs

Revision ID: aa70ce221cc6
Revises: 3bbcbf9e6b23
Create Date: 2026-09-10 10:34:17.943382

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'aa70ce221cc6'
down_revision: Union[str, Sequence[str], None] = '3bbcbf9e6b23'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('course_jobs', sa.Column('created_by_user_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_course_jobs_created_by_user_id'), 'course_jobs', ['created_by_user_id'], unique=False)
    op.create_foreign_key(None, 'course_jobs', 'users', ['created_by_user_id'], ['id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(None, 'course_jobs', type_='foreignkey')
    op.drop_index(op.f('ix_course_jobs_created_by_user_id'), table_name='course_jobs')
    op.drop_column('course_jobs', 'created_by_user_id')
