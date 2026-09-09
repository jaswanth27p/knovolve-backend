from datetime import datetime
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, and_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class Assignment(Base):
    __tablename__ = "assignments"
    id: Mapped[int] = mapped_column(primary_key=True)
    level: Mapped[str] = mapped_column(String(16))  # "chapter" | "module"
    chapter_content_id: Mapped[int | None] = mapped_column(ForeignKey("chapter_contents.id"), nullable=True, index=True)
    module_id: Mapped[int | None] = mapped_column(ForeignKey("modules.id"), nullable=True, index=True)
    scope: Mapped[str] = mapped_column(String(16))  # "global" | "user"
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="generating")  # generating|ready|failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "(level = 'chapter' AND chapter_content_id IS NOT NULL AND module_id IS NULL) OR "
            "(level = 'module' AND module_id IS NOT NULL AND chapter_content_id IS NULL)",
            name="ck_assignments_level_target",
        ),
        Index(
            "ix_assignments_one_global_per_chapter_content",
            "chapter_content_id", unique=True,
            postgresql_where=and_(scope == "global", level == "chapter"),
        ),
        Index(
            "ix_assignments_one_user_per_chapter_content",
            "chapter_content_id", "user_id", unique=True,
            postgresql_where=and_(scope == "user", level == "chapter"),
        ),
        Index(
            "ix_assignments_one_global_per_module",
            "module_id", unique=True,
            postgresql_where=and_(scope == "global", level == "module"),
        ),
        Index(
            "ix_assignments_one_user_per_module",
            "module_id", "user_id", unique=True,
            postgresql_where=and_(scope == "user", level == "module"),
        ),
    )


class AssignmentQuestion(Base):
    __tablename__ = "assignment_questions"
    id: Mapped[int] = mapped_column(primary_key=True)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("assignments.id"), index=True)
    order: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(16))  # "mcq" | "true_false" | "free_text"
    text: Mapped[str] = mapped_column(Text)
    options: Mapped[list | None] = mapped_column(JSONB, nullable=True)  # list[str]; null for true_false/free_text
    correct_answer: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    concept_tag: Mapped[str] = mapped_column(String(255))
    difficulty: Mapped[str] = mapped_column(String(16))  # "easy" | "medium" | "hard"
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    source_section_id: Mapped[int | None] = mapped_column(ForeignKey("chapter_content_sections.id"), nullable=True)

    __table_args__ = (
        Index("ix_assignment_questions_unique_order", "assignment_id", "order", unique=True),
    )


class AssignmentUserTopup(Base):
    __tablename__ = "assignment_user_topups"
    id: Mapped[int] = mapped_column(primary_key=True)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("assignments.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="generating")  # generating|ready|failed|skipped
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("assignment_id", "user_id", name="uq_assignment_user_topup"),
    )
