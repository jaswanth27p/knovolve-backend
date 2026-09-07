"""add user_courses table

Revision ID: 116632f0e3bc
Revises: fb0cc537d4e7
Create Date: 2026-09-08 02:17:30.057105

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '116632f0e3bc'
down_revision: Union[str, Sequence[str], None] = 'fb0cc537d4e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('user_courses',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('course_id', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('progress', sa.Float(), nullable=False),
    sa.Column('enrolled_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_opened_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['course_id'], ['courses.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'course_id', name='uq_user_courses_user_course')
    )
    op.create_index(op.f('ix_user_courses_course_id'), 'user_courses', ['course_id'], unique=False)
    op.create_index(op.f('ix_user_courses_user_id'), 'user_courses', ['user_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_user_courses_user_id'), table_name='user_courses')
    op.drop_index(op.f('ix_user_courses_course_id'), table_name='user_courses')
    op.drop_table('user_courses')
