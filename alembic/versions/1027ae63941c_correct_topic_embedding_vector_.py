"""correct topic_embedding vector dimension to 2048

Revision ID: 1027ae63941c
Revises: 116632f0e3bc
Create Date: 2026-09-08 03:43:34.408115

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '1027ae63941c'
down_revision: Union[str, Sequence[str], None] = '116632f0e3bc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# EMBEDDING_DIM was set to 1024 as an unverified assumption about the
# nemotron-3-embed-1b provider's output dimension; the real endpoint returns
# 2048-d vectors, which pgvector rejects against a vector(1024) column.
# A dimension change isn't a value-preserving cast (old embeddings would be
# meaningless at the new width), so this drops and re-adds the columns rather
# than altering in place -- correct behavior forces re-embedding existing
# rows, not silent data corruption.
def upgrade() -> None:
    op.drop_column("courses", "topic_embedding")
    op.add_column("courses", sa.Column("topic_embedding", Vector(2048), nullable=False))
    op.drop_column("course_jobs", "topic_embedding")
    op.add_column("course_jobs", sa.Column("topic_embedding", Vector(2048), nullable=False))


def downgrade() -> None:
    op.drop_column("courses", "topic_embedding")
    op.add_column("courses", sa.Column("topic_embedding", Vector(1024), nullable=False))
    op.drop_column("course_jobs", "topic_embedding")
    op.add_column("course_jobs", sa.Column("topic_embedding", Vector(1024), nullable=False))
