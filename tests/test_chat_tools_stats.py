from datetime import datetime, timedelta, timezone

from app.db import SessionLocal
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Course, Module
from app.models.user import User
from app.services.chat_tools import stats as stats_tools


def _now():
    return datetime.now(timezone.utc)


def test_get_user_stats_counts_only_todays_attempts():
    with SessionLocal() as db:
        user = User(email="stats-a@example.com", password_hash="x")
        db.add(user); db.flush()
        course = Course(topic_slug="stats-course-a", topic_raw="A", topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course); db.flush()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module); db.flush()
        chapter = Chapter(module_id=module.id, title="C1", objective="o", order=1)
        db.add(chapter); db.flush()
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                  outline=[], created_at=_now(), updated_at=_now())
        db.add(content); db.flush()
        assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                                 status="ready", created_at=_now(), updated_at=_now())
        db.add(assignment); db.flush()
        db.add(AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                  overall_score=1.0, created_at=_now(), updated_at=_now()))
        db.add(AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                  overall_score=1.0, created_at=_now() - timedelta(days=2), updated_at=_now()))
        db.commit()
        user_id = user.id

    with SessionLocal() as db:
        result = stats_tools.get_user_stats(db, user_id)
    assert result["assignments_attempted_today"] == 1
    assert result["streak_current"] == 0
