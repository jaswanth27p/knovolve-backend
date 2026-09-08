from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class AssignmentAttempt(Base):
    __tablename__ = "assignment_attempts"
    id: Mapped[int] = mapped_column(primary_key=True)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("assignments.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="grading")  # grading|graded|failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # fraction correct, null until graded
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AssignmentAnswer(Base):
    __tablename__ = "assignment_answers"
    id: Mapped[int] = mapped_column(primary_key=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("assignment_attempts.id"), index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("assignment_questions.id"))
    concept_tag: Mapped[str] = mapped_column(String(255))  # denormalized copy of the question's tag at grading time
    user_answer: Mapped[str] = mapped_column(Text)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # null until graded
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)  # LLM explanation (free_text) or canned message (mcq/true_false)
    graded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_assignment_answers_unique_per_attempt", "attempt_id", "question_id", unique=True),
    )
