from datetime import datetime, timezone
import pytest
from sqlalchemy.exc import IntegrityError
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection


def _now():
    return datetime.now(timezone.utc)


def _make_chapter(db, topic_slug: str) -> int:
    course = Course(topic_slug=topic_slug, topic_raw=topic_slug, topic_embedding=[0.0] * 1024, created_at=_now())
    db.add(course)
    db.commit()
    module = Module(course_id=course.id, title="M", objective="o", order=1)
    db.add(module)
    db.commit()
    chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
    db.add(chapter)
    db.commit()
    return chapter.id


def test_duplicate_global_version_rejected():
    with SessionLocal() as db:
        chapter_id = _make_chapter(db, "cc-models-dup")
        db.add(ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="generating",
                               outline=[], created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="generating",
                               outline=[], created_at=_now(), updated_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_user_scoped_content_does_not_collide_with_global():
    with SessionLocal() as db:
        chapter_id = _make_chapter(db, "cc-models-user-scope")
        db.add(ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="ready",
                               outline=[], created_at=_now(), updated_at=_now()))
        db.commit()
        # A user-scoped row at the same (chapter_id, version) must NOT collide —
        # the partial unique index only covers scope="global" rows.
        db.add(ChapterContent(chapter_id=chapter_id, version=1, scope="user", user_id=None, status="generating",
                               outline=[], created_at=_now(), updated_at=_now()))
        db.commit()  # should not raise


def test_duplicate_section_order_rejected():
    with SessionLocal() as db:
        chapter_id = _make_chapter(db, "cc-models-dup-order")
        content = ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="generating",
                                  outline=[], created_at=_now(), updated_at=_now())
        db.add(content)
        db.commit()
        db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H", kind="teaching",
                                      body_markdown="b", examples=[{"prompt": "p", "walkthrough": "w"}]))
        db.commit()
        db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H2", kind="teaching",
                                      body_markdown="b2", examples=[{"prompt": "p", "walkthrough": "w"}]))
        with pytest.raises(IntegrityError):
            db.commit()
