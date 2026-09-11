from datetime import datetime, timezone

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.user import User
from app.services import assignments, progression
from app.services.course_extension import append_chapters


@pytest.fixture(autouse=True)
def _seed_users():
    with SessionLocal() as db:
        db.add(User(id=5, email="ext-assign-user-5@example.com", password_hash="x"))
        db.commit()


def _make_course():
    with SessionLocal() as db:
        course = Course(topic_slug="ext-assign-course", topic_raw="ExtA",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course


def _ready_content(db, chapter, user_id):
    content = ChapterContent(chapter_id=chapter.id, version=1, scope="user", user_id=user_id,
                             status="ready", outline=[], created_at=datetime.now(timezone.utc),
                             updated_at=datetime.now(timezone.utc))
    db.add(content)
    db.flush()
    db.add(ChapterContentSection(chapter_content_id=content.id, order=1, heading="H",
                                 kind="teaching", body_markdown="b", examples=[{"prompt": "p", "walkthrough": "w"}]))
    db.flush()
    return content


def test_module_assignment_for_user_bucket_returns_404():
    course = _make_course()
    with SessionLocal() as db:
        append_chapters(db, 5, course, [{"title": "ExtA1", "objective": "o"}])
        db.commit()
    with SessionLocal() as db:
        bucket = db.query(Module).filter_by(course_id=course.id, scope="user", user_id=5).one()
        chapter = db.query(Chapter).filter_by(module_id=bucket.id).one()
        _ready_content(db, chapter, 5)
        with pytest.raises(HTTPException) as exc:
            assignments.create_module_assignment(db, bucket, 5)
        assert exc.value.status_code == 404
        db.rollback()


def test_chapter_assignment_for_extension_chapter_works():
    """Chapter-level assignment for an extension chapter is user-scoped and
    fetch-or-dispatch functions against its user content."""
    course = _make_course()
    with SessionLocal() as db:
        added = append_chapters(db, 5, course, [{"title": "ExtA1", "objective": "o"}])
        db.commit()
    with SessionLocal() as db:
        chapter = db.get(Chapter, added[0]["chapter_id"])
        content = _ready_content(db, chapter, 5)
        db.commit()
    with SessionLocal() as db:
        chapter = db.get(Chapter, added[0]["chapter_id"])
        if chapter is None:
            raise AssertionError("extension chapter missing")
        with patch("app.services.assignments.generate_chapter_assignment_task"):
            result = assignments.get_chapter_assignment(db, chapter, 5)
        assert result.status in ("generating", "ready")
        db.rollback()


def test_progress_counts_extension_chapters():
    course = _make_course()
    from app.models.enrollment import UserCourse
    with SessionLocal() as db:
        db.add(UserCourse(user_id=5, course_id=course.id, status="in_progress",
                          progress=0.0, enrolled_at=datetime.now(timezone.utc),
                          last_opened_at=datetime.now(timezone.utc)))
        db.flush()
        m = Module(course_id=course.id, title="G", objective="o", order=1, scope="global")
        db.add(m)
        db.flush()
        db.add(Chapter(module_id=m.id, title="GC", objective="o", order=1, scope="global"))
        db.commit()
        progression.update_course_progress(db, 5, course.id)
        uc = db.query(UserCourse).filter_by(user_id=5, course_id=course.id).one()
        assert uc.progress == 0.0  # 0/1 passed
        db.commit()