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


@pytest.fixture(autouse=True)
def clean_db():
    yield
    with SessionLocal() as s:
        s.execute(text(
            "TRUNCATE refresh_tokens, learner_streaks, users, assignment_answers, assignment_attempts, "
            "assignment_questions, assignments, chapter_content_sections, chapter_contents, "
            "concept_edges, concepts, chapters, modules, course_jobs, courses, user_courses, "
            "checkpoint_writes, checkpoint_blobs, checkpoints RESTART IDENTITY CASCADE"
        ))
        s.commit()
