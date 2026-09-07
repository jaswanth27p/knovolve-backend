from datetime import datetime, timezone
import pytest
from sqlalchemy.exc import IntegrityError
from app.db import SessionLocal
from app.models.course import Course
from app.models.user import User
from app.models.enrollment import UserCourse


def _now():
    return datetime.now(timezone.utc)


def _make_user_and_course(db, email: str, slug: str) -> tuple[int, int]:
    user = User(email=email, password_hash="x")
    db.add(user)
    db.flush()
    course = Course(topic_slug=slug, topic_raw=slug,
                    topic_embedding=[0.0] * 1024, created_at=_now())
    db.add(course)
    db.flush()
    return user.id, course.id


def test_duplicate_user_course_rejected():
    with SessionLocal() as db:
        user_id, course_id = _make_user_and_course(db, "uc-model-a@example.com", "uc-model-a")
        db.add(UserCourse(user_id=user_id, course_id=course_id, enrolled_at=_now(), last_opened_at=_now()))
        db.commit()
        db.add(UserCourse(user_id=user_id, course_id=course_id, enrolled_at=_now(), last_opened_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_same_course_different_users_allowed():
    with SessionLocal() as db:
        u1, course_id = _make_user_and_course(db, "uc-model-b1@example.com", "uc-model-b")
        u2, _ = _make_user_and_course(db, "uc-model-b2@example.com", "uc-model-b2")
        db.add(UserCourse(user_id=u1, course_id=course_id, enrolled_at=_now(), last_opened_at=_now()))
        db.add(UserCourse(user_id=u2, course_id=course_id, enrolled_at=_now(), last_opened_at=_now()))
        db.commit()  # must not raise


def test_defaults():
    with SessionLocal() as db:
        user_id, course_id = _make_user_and_course(db, "uc-model-c@example.com", "uc-model-c")
        uc = UserCourse(user_id=user_id, course_id=course_id, enrolled_at=_now(), last_opened_at=_now())
        db.add(uc)
        db.commit()
        db.refresh(uc)
        assert uc.status == "in_progress"
        assert uc.progress == 0.0