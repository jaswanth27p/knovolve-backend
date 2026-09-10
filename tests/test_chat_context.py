from datetime import datetime, timezone

from app.db import SessionLocal
from app.models.course import Chapter, Course, Module
from app.models.enrollment import UserCourse
from app.models.user import User
from app.schemas.chat import RouteContext
from app.services.chat_context import build_context_bundle


def _now():
    return datetime.now(timezone.utc)


def test_bundle_with_no_route_has_no_current_course():
    with SessionLocal() as db:
        user = User(email="ctx-a@example.com", password_hash="x")
        db.add(user); db.commit()
        bundle = build_context_bundle(db, user.id, None)
    assert bundle.current_course is None
    assert bundle.current_course_modules is None


def test_bundle_with_route_to_started_course_includes_modules_and_concepts():
    with SessionLocal() as db:
        user = User(email="ctx-b@example.com", password_hash="x")
        db.add(user); db.flush()
        course = Course(topic_slug="ctx-course-b", topic_raw="Ctx Course B",
                         topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course); db.flush()
        module = Module(course_id=course.id, title="M1", objective="o", order=1)
        db.add(module); db.flush()
        db.add(Chapter(module_id=module.id, title="C1", objective="o", order=1))
        db.add(UserCourse(user_id=user.id, course_id=course.id, enrolled_at=_now(), last_opened_at=_now()))
        db.commit()
        user_id, slug = user.id, course.topic_slug

    with SessionLocal() as db:
        bundle = build_context_bundle(db, user_id, RouteContext(page="course", course_slug=slug))

    assert bundle.current_course is not None
    assert bundle.current_course.topic_slug == slug
    assert bundle.current_course.course_url.endswith(f"/courses/{slug}")
    assert bundle.current_course_modules is not None
    assert bundle.current_course_modules[0].chapters[0].completed is False


def test_bundle_with_route_to_not_started_course_has_summary_but_no_modules():
    with SessionLocal() as db:
        user = User(email="ctx-c@example.com", password_hash="x")
        db.add(user); db.flush()
        course = Course(topic_slug="ctx-course-c", topic_raw="Ctx Course C",
                         topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course); db.commit()
        user_id, slug = user.id, course.topic_slug

    with SessionLocal() as db:
        bundle = build_context_bundle(db, user_id, RouteContext(page="course", course_slug=slug))

    assert bundle.current_course is not None
    assert bundle.current_course.status == "not_started"
    assert bundle.current_course_modules is None
