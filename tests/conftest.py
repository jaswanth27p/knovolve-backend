import pytest
from sqlalchemy import text
from app.db import SessionLocal
# Import all models to register them with SQLAlchemy's Base metadata
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.models.course import Course, CourseJob, Module, Chapter, Concept, ConceptEdge
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.enrollment import UserCourse
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt, AssignmentAnswer
from app.models.learner_streak import LearnerStreak


class _NullRedis:
    """Stands in for Redis in tests: reads miss, writes succeed silently.
    Course preview caching is convenience-only (missing entries degrade to
    recomputation), so a null client keeps the suite hermetic."""
    def get(self, _):
        return None

    def set(self, *_args, **_kwargs):
        return True

    def delete(self, *_args):
        return 1


@pytest.fixture(autouse=True)
def _no_redis(monkeypatch):
    from app.services import course_preview
    monkeypatch.setattr(course_preview, "_get_client", lambda: _NullRedis())


@pytest.fixture(autouse=True)
def clean_db():
    yield
    with SessionLocal() as s:
        s.execute(text(
            "TRUNCATE refresh_tokens, learner_streaks, users, assignment_answers, assignment_attempts, "
            "assignment_questions, assignment_user_topups, assignments, chapter_content_sections, chapter_contents, "
            "concept_edges, concepts, chapters, modules, course_jobs, courses, user_courses, "
            "checkpoint_writes, checkpoint_blobs, checkpoints RESTART IDENTITY CASCADE"
        ))
        s.commit()
