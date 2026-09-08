from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Integer, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class ChapterContent(Base):
    __tablename__ = "chapter_contents"
    id: Mapped[int] = mapped_column(primary_key=True)
    chapter_id: Mapped[int] = mapped_column(ForeignKey("chapters.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    scope: Mapped[str] = mapped_column(String(16))  # "global" | "user" (only "global" used so far)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="generating")  # generating|ready|failed
    # The section outline (list[{heading, objective, kind}]), persisted once
    # right after generate_section_outline runs so a crash mid-run resumes
    # against the SAME planned sections instead of re-planning (which could
    # produce a different heading/order set and desync already-persisted
    # ChapterContentSection rows from their `order` index).
    outline: Mapped[list] = mapped_column(JSONB, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    remediation_target_tags: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # list[str]; None for V1, the exact remediation_concept_tags that triggered this version for V2+
    remediation_source_attempt_id: Mapped[int | None] = mapped_column(ForeignKey("assignment_attempts.id"), nullable=True)  # which graded attempt's verdict triggered this version; None for V1
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index(
            "ix_chapter_contents_one_global_per_chapter",
            "chapter_id", "version",
            unique=True,
            postgresql_where=(scope == "global"),
        ),
        Index(
            "ix_chapter_contents_one_user_version_per_chapter",
            "chapter_id", "user_id", "version",
            unique=True,
            postgresql_where=(scope == "user"),
        ),
        Index(
            "ix_chapter_contents_unique_remediation_source",
            "remediation_source_attempt_id",
            unique=True,
            postgresql_where=remediation_source_attempt_id.isnot(None),
        ),
    )


class ChapterContentSection(Base):
    __tablename__ = "chapter_content_sections"
    id: Mapped[int] = mapped_column(primary_key=True)
    chapter_content_id: Mapped[int] = mapped_column(ForeignKey("chapter_contents.id"), index=True)
    order: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(16), default="teaching")  # "intro" | "teaching"
    body_markdown: Mapped[str] = mapped_column(Text)
    examples: Mapped[list] = mapped_column(JSONB, default=list)  # list[{prompt, walkthrough}]
    diagram_spec: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # {nodes, edges} or null
    diagram_status: Mapped[str | None] = mapped_column(String(16), nullable=True)  # pending|ready|failed
    diagram_image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    diagram_attempts: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("ix_chapter_content_sections_unique_order", "chapter_content_id", "order", unique=True),
    )
