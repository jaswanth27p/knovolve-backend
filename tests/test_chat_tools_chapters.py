from datetime import datetime, timezone

import pytest

from app.db import SessionLocal
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter, Course, Module
from app.models.enrollment import UserCourse
from app.models.user import User
from app.services.chat_tools import chapters as chapter_tools
from app.services.chat_tools._errors import ChatToolError


def _now():
    return datetime.now(timezone.utc)


def _make_started_chapter(db, slug: str, user_id: int):
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
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
    db.add(UserCourse(user_id=user_id, course_id=course.id, enrolled_at=_now(), last_opened_at=_now()))
    db.flush()
    return course, chapter, content, assignment


def test_get_chapter_progress_reports_attempt_count_and_completion():
    with SessionLocal() as db:
        user = User(email="chp-a@example.com", password_hash="x")
        db.add(user); db.flush()
        course, chapter, content, assignment = _make_started_chapter(db, "chp-course-a", user.id)
        db.add(AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                  overall_score=0.9, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, slug, chapter_id = user.id, course.topic_slug, chapter.id

    with SessionLocal() as db:
        result = chapter_tools.get_chapter_progress(db, user_id, slug, chapter_id)
    assert result["completed"] is True
    assert result["attempt_count"] == 1
    assert result["latest_score"] == 0.9
    assert result["content_version_count"] == 1


def test_get_chapter_content_returns_unavailable_when_not_ready():
    with SessionLocal() as db:
        user = User(email="chp-b@example.com", password_hash="x")
        db.add(user); db.flush()
        course = Course(topic_slug="chp-course-b", topic_raw="B", topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course); db.flush()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module); db.flush()
        chapter = Chapter(module_id=module.id, title="C1", objective="o", order=1)
        db.add(chapter); db.flush()
        db.add(ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="generating",
                               outline=[], created_at=_now(), updated_at=_now()))
        db.add(UserCourse(user_id=user.id, course_id=course.id, enrolled_at=_now(), last_opened_at=_now()))
        db.commit()
        user_id, slug, chapter_id = user.id, course.topic_slug, chapter.id

    with SessionLocal() as db:
        result = chapter_tools.get_chapter_content(db, user_id, slug, chapter_id)
    assert result["available"] is False


def test_get_chapter_content_returns_sections_when_ready():
    with SessionLocal() as db:
        user = User(email="chp-c@example.com", password_hash="x")
        db.add(user); db.flush()
        course, chapter, content, assignment = _make_started_chapter(db, "chp-course-c", user.id)
        db.add(ChapterContentSection(chapter_content_id=content.id, order=1, heading="Intro",
                                      kind="intro", body_markdown="Hello world", examples=[]))
        db.commit()
        user_id, slug, chapter_id = user.id, course.topic_slug, chapter.id

    with SessionLocal() as db:
        result = chapter_tools.get_chapter_content(db, user_id, slug, chapter_id)
    assert result["available"] is True
    assert result["sections"] == [{"heading": "Intro", "body_markdown": "Hello world"}]


def _add_user_version(db, chapter_id: int, user_id: int, version: int, status="ready"):
    content = ChapterContent(chapter_id=chapter_id, version=version, scope="user", user_id=user_id,
                              status=status, outline=[], created_at=_now(), updated_at=_now())
    db.add(content); db.flush()
    return content


def test_get_chapter_content_fetches_specific_version():
    with SessionLocal() as db:
        user = User(email="chp-d@example.com", password_hash="x")
        db.add(user); db.flush()
        course, chapter, content_v1, _assignment = _make_started_chapter(db, "chp-course-d", user.id)
        db.add(ChapterContentSection(chapter_content_id=content_v1.id, order=1, heading="V1 head",
                                      kind="teaching", body_markdown="V1 body", examples=[]))
        content_v2 = _add_user_version(db, chapter.id, user.id, version=2)
        db.add(ChapterContentSection(chapter_content_id=content_v2.id, order=1, heading="V2 head",
                                      kind="teaching", body_markdown="V2 body", examples=[]))
        db.commit()
        user_id, slug, chapter_id = user.id, course.topic_slug, chapter.id

    with SessionLocal() as db:
        result_v1 = chapter_tools.get_chapter_content(db, user_id, slug, chapter_id, version=1)
        result_v2 = chapter_tools.get_chapter_content(db, user_id, slug, chapter_id, version=2)
    assert result_v1["sections"] == [{"heading": "V1 head", "body_markdown": "V1 body"}]
    assert result_v2["sections"] == [{"heading": "V2 head", "body_markdown": "V2 body"}]


def test_get_chapter_content_omitted_version_resolves_to_latest_user_content():
    with SessionLocal() as db:
        user = User(email="chp-e@example.com", password_hash="x")
        db.add(user); db.flush()
        course, chapter, content_v1, _assignment = _make_started_chapter(db, "chp-course-e", user.id)
        _add_user_version(db, chapter.id, user.id, version=2)
        db.commit()
        user_id, slug, chapter_id = user.id, course.topic_slug, chapter.id

    with SessionLocal() as db:
        result = chapter_tools.get_chapter_content(db, user_id, slug, chapter_id)
    assert result["available"] is True
    assert result["version"] == 2


def test_get_chapter_content_rejects_other_users_remediation_version():
    with SessionLocal() as db:
        owner = User(email="chp-f@example.com", password_hash="x")
        other = User(email="chp-g@example.com", password_hash="x")
        db.add_all([owner, other]); db.flush()
        course, chapter, _content_v1, _assignment = _make_started_chapter(db, "chp-course-f", owner.id)
        _add_user_version(db, chapter.id, other.id, version=2)
        db.commit()
        owner_id, slug, chapter_id = owner.id, course.topic_slug, chapter.id

    with SessionLocal() as db:
        with pytest.raises(ChatToolError, match="No version 2"):
            chapter_tools.get_chapter_content(db, owner_id, slug, chapter_id, version=2)


def test_get_chapter_content_not_ready_version_returns_unavailable():
    with SessionLocal() as db:
        user = User(email="chp-h@example.com", password_hash="x")
        db.add(user); db.flush()
        course, chapter, _content_v1, _assignment = _make_started_chapter(db, "chp-course-h", user.id)
        _add_user_version(db, chapter.id, user.id, version=2, status="generating")
        db.commit()
        user_id, slug, chapter_id = user.id, course.topic_slug, chapter.id

    with SessionLocal() as db:
        result = chapter_tools.get_chapter_content(db, user_id, slug, chapter_id, version=2)
    assert result["available"] is False
