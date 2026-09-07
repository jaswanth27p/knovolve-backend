import pytest
from sqlalchemy import text
from app.db import SessionLocal
# Import all models to register them with SQLAlchemy's Base metadata
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.models.course import Course, CourseJob, Module, Chapter, Concept, ConceptEdge
from app.models.chapter_content import ChapterContent, ChapterContentSection


@pytest.fixture(autouse=True)
def clean_db():
    yield
    with SessionLocal() as s:
        s.execute(text(
            "TRUNCATE refresh_tokens, users, chapter_content_sections, chapter_contents, "
            "concept_edges, concepts, chapters, modules, course_jobs, courses, "
            "checkpoint_writes, checkpoint_blobs, checkpoints RESTART IDENTITY CASCADE"
        ))
        s.commit()
