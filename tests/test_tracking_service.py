from datetime import datetime, timezone

import pytest
from sqlalchemy import event

from app.db import SessionLocal, engine
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Concept, Course, Module
from app.models.enrollment import UserCourse
from app.models.user import User
from app.services import tracking


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def query_counter():
    """Counts every SQL statement the engine actually executes. Used to prove
    the tracked-course serializers are constant-query rather than N+1."""
    counts = {"n": 0}

    def _before(*_args, **_kwargs):
        counts["n"] += 1

    event.listen(engine, "before_cursor_execute", _before)
    try:
        yield counts
    finally:
        event.remove(engine, "before_cursor_execute", _before)


def _seed_user(n_courses: int, suffix: str, chapters_per_module: int = 2,
               content_ready: bool = False) -> int:
    with SessionLocal() as db:
        user = User(email=f"tracking-{suffix}@example.com", password_hash="x")
        db.add(user)
        db.flush()
        for i in range(n_courses):
            course = Course(topic_slug=f"tracking-{suffix}-{i}", topic_raw=f"tracking-{suffix}-{i}",
                            topic_embedding=[0.0] * 2048, created_at=_now())
            db.add(course)
            db.flush()
            for m_i in range(2):
                module = Module(course_id=course.id, title=f"M{m_i}", objective="o", order=m_i)
                db.add(module)
                db.flush()
                for c_i in range(chapters_per_module):
                    chapter = Chapter(module_id=module.id, title=f"C{c_i}", objective="o", order=c_i)
                    db.add(chapter)
                    db.flush()
                    db.add(Concept(course_id=course.id, name=f"concept-{suffix}-{i}-{m_i}-{c_i}",
                                   chapter_id=chapter.id))
                    if content_ready:
                        db.add(ChapterContent(chapter_id=chapter.id, version=1, scope="global",
                                              status="ready", outline=[],
                                              created_at=_now(), updated_at=_now()))
            db.add(UserCourse(user_id=user.id, course_id=course.id, status="in_progress", progress=0.0,
                              enrolled_at=_now(), last_opened_at=_now()))
        db.commit()
        return user.id


def test_list_my_courses_query_count_does_not_scale_with_course_count(query_counter):
    uid_small = _seed_user(2, "lc-small")
    uid_big = _seed_user(4, "lc-big")

    with SessionLocal() as db:
        query_counter["n"] = 0
        rows, total = tracking.list_my_courses(db, uid_small, limit=100)
        small = query_counter["n"]
    assert total == 2 and len(rows) == 2

    with SessionLocal() as db:
        query_counter["n"] = 0
        rows, total = tracking.list_my_courses(db, uid_big, limit=100)
        big = query_counter["n"]
    assert total == 4 and len(rows) == 4

    assert small == big, f"query count scaled with course count: {small} -> {big}"
    assert small < 30


def test_get_dashboard_query_count_does_not_scale_with_course_count(query_counter):
    uid_small = _seed_user(2, "db-small")
    uid_big = _seed_user(4, "db-big")

    with SessionLocal() as db:
        query_counter["n"] = 0
        dash = tracking.get_dashboard(db, uid_small)
        small = query_counter["n"]
    assert dash.total_count == 2

    with SessionLocal() as db:
        query_counter["n"] = 0
        dash = tracking.get_dashboard(db, uid_big)
        big = query_counter["n"]
    assert dash.total_count == 4

    assert small == big, f"query count scaled with course count: {small} -> {big}"
    assert small < 30


def test_dashboard_caps_serialized_rows_but_reports_true_totals(monkeypatch):
    monkeypatch.setattr(tracking, "DASHBOARD_COURSE_LIMIT", 2)
    uid = _seed_user(4, "cap")  # 4 in_progress enrollments, cap is 2

    with SessionLocal() as db:
        dash = tracking.get_dashboard(db, uid)

    assert len(dash.in_progress) == 2
    assert len(dash.completed) == 0
    assert dash.in_progress_count == 4
    assert dash.completed_count == 0
    assert dash.total_count == 4


def test_serialize_tracked_excludes_empty_user_module_and_reports_ready():
    with SessionLocal() as db:
        user = User(email="tracking-empty-mod@example.com", password_hash="x")
        db.add(user)
        db.flush()
        course = Course(topic_slug="tracking-empty-mod", topic_raw="tracking-empty-mod",
                        topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course)
        db.flush()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.flush()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
        db.add(chapter)
        db.flush()
        db.add(ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                              outline=[], created_at=_now(), updated_at=_now()))
        # Empty user-scope extension bucket: must be visible but excluded.
        db.add(Module(course_id=course.id, title="Ext", objective="o", order=2,
                      scope="user", user_id=user.id))
        db.add(UserCourse(user_id=user.id, course_id=course.id, status="in_progress", progress=0.0,
                          enrolled_at=_now(), last_opened_at=_now()))
        db.commit()
        user_id = user.id

    with SessionLocal() as db:
        rows, _total = tracking.list_my_courses(db, user_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.module_count == 1
    assert row.chapter_count == 1
    assert row.content_ready is True


def test_get_tracked_course_by_slug_scopes_to_user_and_course():
    uid = _seed_user(2, "slug")
    other = _seed_user(1, "slug-other")

    with SessionLocal() as db:
        row = tracking.get_tracked_course_by_slug(db, uid, "tracking-slug-0")
        assert row is not None
        assert row.topic_slug == "tracking-slug-0"
        # another user's enrollment for the same slug is not visible
        assert tracking.get_tracked_course_by_slug(db, other, "tracking-slug-0") is None
        # this user's untracked slug is not visible
        assert tracking.get_tracked_course_by_slug(db, uid, "tracking-slug-other-0") is None
