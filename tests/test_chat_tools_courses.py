from datetime import datetime, timezone
from unittest.mock import patch

from app.db import SessionLocal
from app.models.course import Chapter, Course, Module
from app.models.enrollment import UserCourse
from app.models.user import User
from app.services.chat_tools import courses as course_tools
from app.services.chat_tools._errors import ChatToolError


def _now():
    return datetime.now(timezone.utc)


def test_browse_public_courses_excludes_started():
    with SessionLocal() as db:
        user = User(email="ct-a@example.com", password_hash="x")
        db.add(user); db.flush()
        started = Course(topic_slug="ct-started", topic_raw="Started",
                          topic_embedding=[0.0] * 2048, created_at=_now())
        not_started = Course(topic_slug="ct-not-started", topic_raw="Not Started",
                              topic_embedding=[0.0] * 2048, created_at=_now())
        db.add_all([started, not_started]); db.flush()
        db.add(UserCourse(user_id=user.id, course_id=started.id, enrolled_at=_now(), last_opened_at=_now()))
        db.commit()
        user_id = user.id

    with SessionLocal() as db:
        result = course_tools.browse_public_courses(db, user_id)
    slugs = {r["topic_slug"] for r in result}
    assert "ct-not-started" in slugs
    assert "ct-started" not in slugs
    assert result[0]["course_url"].endswith(f"/courses/{result[0]['topic_slug']}")


def test_search_courses_ranks_by_embedding_similarity():
    with SessionLocal() as db:
        close = Course(topic_slug="ct-search-close", topic_raw="Close",
                        topic_embedding=[1.0] + [0.0] * 2047, created_at=_now())
        far = Course(topic_slug="ct-search-far", topic_raw="Far",
                      topic_embedding=[0.0, 1.0] + [0.0] * 2046, created_at=_now())
        db.add_all([close, far]); db.commit()

    with patch("app.services.chat_tools.courses.embed", return_value=[1.0] + [0.0] * 2047):
        with SessionLocal() as db:
            result = course_tools.search_courses(db, user_id=1, query="close topic", limit=2)
    assert result[0]["topic_slug"] == "ct-search-close"


def test_get_course_modules_requires_started_course():
    with SessionLocal() as db:
        course = Course(topic_slug="ct-modules-a", topic_raw="Modules A",
                         topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course); db.commit()

    with SessionLocal() as db:
        try:
            course_tools.get_course_modules(db, user_id=999, course_slug="ct-modules-a")
            assert False, "expected ChatToolError"
        except ChatToolError as exc:
            assert "haven't started" in str(exc)


def test_get_course_modules_marks_completion():
    with SessionLocal() as db:
        user = User(email="ct-b@example.com", password_hash="x")
        db.add(user); db.flush()
        course = Course(topic_slug="ct-modules-b", topic_raw="Modules B",
                         topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course); db.flush()
        module = Module(course_id=course.id, title="M1", objective="o", order=1)
        db.add(module); db.flush()
        db.add(Chapter(module_id=module.id, title="C1", objective="o", order=1))
        db.add(UserCourse(user_id=user.id, course_id=course.id, enrolled_at=_now(), last_opened_at=_now()))
        db.commit()
        user_id = user.id

    with SessionLocal() as db:
        result = course_tools.get_course_modules(db, user_id, "ct-modules-b")
    assert result[0]["chapters"][0]["completed"] is False
