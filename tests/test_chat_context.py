from datetime import datetime, timezone

import pytest
from sqlalchemy import event

from app.db import SessionLocal, engine
from app.models.course import Chapter, Course, Module
from app.models.enrollment import UserCourse
from app.models.user import User
from app.schemas.chat import RouteContext
from app.services.chat_context import build_context_bundle


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def query_counter():
    counts = {"n": 0}

    def _before(*_args, **_kwargs):
        counts["n"] += 1

    event.listen(engine, "before_cursor_execute", _before)
    try:
        yield counts
    finally:
        event.remove(engine, "before_cursor_execute", _before)


def _seed_active_course(email: str, slug: str, chapters_per_module: int) -> int:
    with SessionLocal() as db:
        user = User(email=email, password_hash="x")
        db.add(user); db.flush()
        course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048,
                         created_at=_now())
        db.add(course); db.flush()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module); db.flush()
        for i in range(chapters_per_module):
            db.add(Chapter(module_id=module.id, title=f"C{i}", objective="o", order=i))
        db.add(UserCourse(user_id=user.id, course_id=course.id, enrolled_at=_now(),
                          last_opened_at=_now()))
        db.commit()
        return user.id


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


def test_build_context_bundle_query_count_does_not_scale_with_chapters(query_counter):
    uid_small = _seed_active_course("ctx-perf-small@example.com", "ctx-perf-small", 2)
    uid_big = _seed_active_course("ctx-perf-big@example.com", "ctx-perf-big", 8)

    with SessionLocal() as db:
        query_counter["n"] = 0
        bundle = build_context_bundle(
            db, uid_small, RouteContext(page="course", course_slug="ctx-perf-small")
        )
        small = query_counter["n"]
    assert bundle.current_course_modules is not None
    assert len(bundle.current_course_modules[0].chapters) == 2

    with SessionLocal() as db:
        query_counter["n"] = 0
        bundle = build_context_bundle(
            db, uid_big, RouteContext(page="course", course_slug="ctx-perf-big")
        )
        big = query_counter["n"]
    assert bundle.current_course_modules is not None
    assert len(bundle.current_course_modules[0].chapters) == 8

    assert small == big, f"query count scaled with chapter count: {small} -> {big}"
    assert small < 40
