from datetime import datetime, timezone

import pytest

from app.db import SessionLocal
from app.models.course import Course
from app.models.enrollment import UserCourse
from app.models.user import User
from app.services.chat_tools._authz import require_started_course
from app.services.chat_tools._errors import ChatToolError


def _now():
    return datetime.now(timezone.utc)


def test_require_started_course_raises_for_unknown_slug():
    with SessionLocal() as db:
        with pytest.raises(ChatToolError, match="No course found"):
            require_started_course(db, user_id=1, course_slug="does-not-exist")


def test_require_started_course_raises_if_not_enrolled():
    with SessionLocal() as db:
        user = User(email="authz-a@example.com", password_hash="x")
        db.add(user); db.flush()
        course = Course(topic_slug="authz-course-a", topic_raw="Authz A",
                         topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course); db.commit()
        user_id = user.id

    with SessionLocal() as db:
        with pytest.raises(ChatToolError, match="haven't started"):
            require_started_course(db, user_id, "authz-course-a")


def test_require_started_course_returns_course_when_enrolled():
    with SessionLocal() as db:
        user = User(email="authz-b@example.com", password_hash="x")
        db.add(user); db.flush()
        course = Course(topic_slug="authz-course-b", topic_raw="Authz B",
                         topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course); db.flush()
        db.add(UserCourse(user_id=user.id, course_id=course.id, enrolled_at=_now(), last_opened_at=_now()))
        db.commit()
        user_id = user.id

    with SessionLocal() as db:
        found = require_started_course(db, user_id, "authz-course-b")
    assert found.topic_slug == "authz-course-b"
