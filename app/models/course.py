from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Integer, Text, Index
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from app.db import Base

EMBEDDING_DIM = 2048  # verified against the live nemotron-3-embed-1b endpoint response


class Course(Base):
    __tablename__ = "courses"
    id: Mapped[int] = mapped_column(primary_key=True)
    topic_slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    topic_raw: Mapped[str] = mapped_column(Text)
    topic_embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    status: Mapped[str] = mapped_column(String(32), default="published")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CourseJob(Base):
    __tablename__ = "course_jobs"
    id: Mapped[int] = mapped_column(primary_key=True)
    topic_slug: Mapped[str] = mapped_column(String(255), index=True)
    topic_raw: Mapped[str] = mapped_column(Text)
    topic_embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|running|succeeded|failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    course_id: Mapped[int | None] = mapped_column(ForeignKey("courses.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index(
            "ix_course_jobs_active_topic_slug",
            "topic_slug",
            unique=True,
            postgresql_where=(status.in_(["pending", "running"])),
        ),
    )


class Module(Base):
    __tablename__ = "modules"
    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    objective: Mapped[str] = mapped_column(Text)
    order: Mapped[int] = mapped_column(Integer)


class Chapter(Base):
    __tablename__ = "chapters"
    id: Mapped[int] = mapped_column(primary_key=True)
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    objective: Mapped[str] = mapped_column(Text)
    order: Mapped[int] = mapped_column(Integer)


class Concept(Base):
    __tablename__ = "concepts"
    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    chapter_id: Mapped[int] = mapped_column(ForeignKey("chapters.id"), index=True)


class ConceptEdge(Base):
    __tablename__ = "concept_edges"
    id: Mapped[int] = mapped_column(primary_key=True)
    concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"), index=True)
    prerequisite_concept_id: Mapped[int] = mapped_column(ForeignKey("concepts.id"), index=True)
