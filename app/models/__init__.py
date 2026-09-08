"""Import every model module so all mapped classes register on Base.metadata
regardless of entry point (FastAPI app, Celery worker, Alembic, a script).

Without this, a process that only imports the specific models it directly
uses can hit SQLAlchemy's NoReferencedTableError on a ForeignKey to a model
class that was never imported elsewhere in its import chain (e.g. the Celery
worker generating assignments never otherwise imports User, so
Assignment.user_id's FK to "users" fails to resolve).
"""
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.models.course import Course, CourseJob, Module, Chapter, Concept, ConceptEdge
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt, AssignmentAnswer
from app.models.enrollment import UserCourse
from app.models.learner_streak import LearnerStreak

__all__ = [
    "User",
    "RefreshToken",
    "Course",
    "CourseJob",
    "Module",
    "Chapter",
    "Concept",
    "ConceptEdge",
    "ChapterContent",
    "ChapterContentSection",
    "Assignment",
    "AssignmentQuestion",
    "AssignmentAttempt",
    "AssignmentAnswer",
    "UserCourse",
    "LearnerStreak",
]
