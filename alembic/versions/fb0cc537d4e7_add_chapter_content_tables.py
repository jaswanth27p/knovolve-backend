"""add chapter content tables

Revision ID: fb0cc537d4e7
Revises: d11f8c188659
Create Date: 2026-09-07 23:49:48.113086

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'fb0cc537d4e7'
down_revision: Union[str, Sequence[str], None] = 'd11f8c188659'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chapter_contents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("chapter_id", sa.Integer(), sa.ForeignKey("chapters.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="generating"),
        sa.Column("outline", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chapter_contents_chapter_id", "chapter_contents", ["chapter_id"])
    op.create_index(
        "ix_chapter_contents_one_global_per_chapter",
        "chapter_contents", ["chapter_id", "version"],
        unique=True, postgresql_where=sa.text("scope = 'global'"),
    )

    op.create_table(
        "chapter_content_sections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("chapter_content_id", sa.Integer(), sa.ForeignKey("chapter_contents.id"), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("heading", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False, server_default="teaching"),
        sa.Column("body_markdown", sa.Text(), nullable=False),
        sa.Column("examples", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("diagram_spec", postgresql.JSONB(), nullable=True),
        sa.Column("diagram_status", sa.String(length=16), nullable=True),
        sa.Column("diagram_image_url", sa.String(length=512), nullable=True),
        sa.Column("diagram_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "ix_chapter_content_sections_chapter_content_id",
        "chapter_content_sections", ["chapter_content_id"],
    )
    op.create_index(
        "ix_chapter_content_sections_unique_order",
        "chapter_content_sections", ["chapter_content_id", "order"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("chapter_content_sections")
    op.drop_table("chapter_contents")
