from datetime import datetime, timezone
import pytest
from sqlalchemy.exc import IntegrityError
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt
from app.models.user import User


def _now():
    return datetime.now(timezone.utc)


def _make_chapter(db, topic_slug: str) -> int:
    course = Course(topic_slug=topic_slug, topic_raw=topic_slug, topic_embedding=[0.0] * 2048, created_at=_now())
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


def _make_attempt(db, chapter_id: int, email: str) -> int:
    """Minimal ready chapter-level global assignment + one graded attempt.
    Returns the attempt id — used only as a foreign key target for
    remediation_source_attempt_id in the tests below."""
    content = ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="ready",
                              outline=[], created_at=_now(), updated_at=_now())
    db.add(content)
    db.commit()
    assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                             status="ready", created_at=_now(), updated_at=_now())
    db.add(assignment)
    db.commit()
    question = AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q",
                                   options=["a", "b"], correct_answer="a", explanation="e",
                                   concept_tag="t1", difficulty="easy")
    db.add(question)
    db.commit()
    user = User(email=email, password_hash="x")
    db.add(user)
    db.commit()
    attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                 overall_score=0.0, created_at=_now(), updated_at=_now())
    db.add(attempt)
    db.commit()
    return attempt.id


def test_remediation_columns_default_null_for_v1():
    with SessionLocal() as db:
        chapter_id = _make_chapter(db, "cc-models-remediation-null")
        content = ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="ready",
                                  outline=[], created_at=_now(), updated_at=_now())
        db.add(content)
        db.commit()
        db.refresh(content)
        assert content.remediation_target_tags is None
        assert content.remediation_source_attempt_id is None


def test_duplicate_user_scoped_version_rejected():
    with SessionLocal() as db:
        chapter_id = _make_chapter(db, "cc-models-dup-user-version")
        user = User(email="cc-models-dup-user-version@example.com", password_hash="x")
        db.add(user)
        db.commit()
        db.add(ChapterContent(chapter_id=chapter_id, version=2, scope="user", user_id=user.id,
                               status="generating", outline=[], created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(ChapterContent(chapter_id=chapter_id, version=2, scope="user", user_id=user.id,
                               status="generating", outline=[], created_at=_now(), updated_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_user_scoped_versions_for_different_users_do_not_collide():
    with SessionLocal() as db:
        chapter_id = _make_chapter(db, "cc-models-multi-user-version")
        user_a = User(email="cc-models-multi-user-a@example.com", password_hash="x")
        user_b = User(email="cc-models-multi-user-b@example.com", password_hash="x")
        db.add(user_a)
        db.add(user_b)
        db.commit()
        db.add(ChapterContent(chapter_id=chapter_id, version=2, scope="user", user_id=user_a.id,
                               status="generating", outline=[], created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(ChapterContent(chapter_id=chapter_id, version=2, scope="user", user_id=user_b.id,
                               status="generating", outline=[], created_at=_now(), updated_at=_now()))
        db.commit()  # should not raise — different users, same version number


def test_duplicate_remediation_source_attempt_rejected():
    with SessionLocal() as db:
        chapter_id = _make_chapter(db, "cc-models-dup-source-attempt")
        attempt_id = _make_attempt(db, chapter_id, "cc-models-dup-source-attempt@example.com")
        db.add(ChapterContent(chapter_id=chapter_id, version=2, scope="user", user_id=None,
                               status="generating", outline=[], remediation_source_attempt_id=attempt_id,
                               created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(ChapterContent(chapter_id=chapter_id, version=3, scope="user", user_id=None,
                               status="generating", outline=[], remediation_source_attempt_id=attempt_id,
                               created_at=_now(), updated_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_multiple_null_remediation_source_rows_allowed():
    with SessionLocal() as db:
        chapter_id = _make_chapter(db, "cc-models-multi-null-source")
        db.add(ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="ready",
                               outline=[], created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(ChapterContent(chapter_id=chapter_id, version=1, scope="user", user_id=None,
                               status="generating", outline=[], created_at=_now(), updated_at=_now()))
        db.commit()  # should not raise — the partial index only applies where remediation_source_attempt_id IS NOT NULL
