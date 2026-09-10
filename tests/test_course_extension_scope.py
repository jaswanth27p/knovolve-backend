import pytest

from datetime import datetime, timezone

from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.user import User
from app.services import courses
from app.services.course_extension import get_or_create_bucket, append_chapters


@pytest.fixture(autouse=True)
def _seed_users():
    with SessionLocal() as db:
        db.add(User(id=5, email="ext-scope-user-5@example.com", password_hash="x"))
        db.add(User(id=6, email="ext-scope-user-6@example.com", password_hash="x"))
        db.commit()


def _make_course():
    with SessionLocal() as db:
        course = Course(topic_slug="ext-scope-course", topic_raw="Scope",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.flush()
        m = Module(course_id=course.id, title="G1", objective="o", order=1, scope="global")
        db.add(m)
        db.flush()
        db.add(Chapter(module_id=m.id, title="IntroC", objective="o", order=1, scope="global"))
        db.commit()
        db.refresh(course)
        return course


def test_serialize_includes_bucket_only_when_nonempty_for_owner():
    course = _make_course()
    with SessionLocal() as db:
        empty = courses.serialize_course(db, course, user_id=5)
        assert [m["title"] for m in empty["modules"]] == ["G1"]
        append_chapters(db, 5, course, [{"title": "Ext1", "objective": "o"}])
        db.commit()
        with_owner = courses.serialize_course(db, course, user_id=5)
        assert [m["title"] for m in with_owner["modules"]] == ["G1", "Additional Chapters"]
        bucket = with_owner["modules"][1]
        assert bucket["is_additional"] is True
        assert [c["title"] for c in bucket["chapters"]] == ["Ext1"]
        other = courses.serialize_course(db, course, user_id=6)
        assert [m["title"] for m in other["modules"]] == ["G1"]


def test_get_module_visible_to_owner_only():
    course = _make_course()
    with SessionLocal() as db:
        bucket = get_or_create_bucket(db, 5, course)
        db.commit()
        m = courses.get_module(db, course, bucket.id, user_id=5)
        assert m.id == bucket.id
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            courses.get_module(db, course, bucket.id, user_id=6)
        db.rollback()


def test_next_chapter_ends_after_bucket():
    course = _make_course()
    with SessionLocal() as db:
        g_module = db.query(Module).filter_by(course_id=course.id, scope="global").one()
        g_chapter = db.query(Chapter).filter_by(module_id=g_module.id).one()
        added = append_chapters(db, 5, course, [{"title": "Ext1", "objective": "o"}])
        db.commit()
        next_id = courses.get_next_chapter_id(db, g_chapter.id, user_id=5)
        assert next_id == added[0]["chapter_id"]
        last = courses.get_next_chapter_id(db, added[0]["chapter_id"], user_id=5)
        assert last is None
        # another user does not descend into the bucket
        assert courses.get_next_chapter_id(db, g_chapter.id, user_id=6) is None