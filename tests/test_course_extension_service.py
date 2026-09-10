from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.db import SessionLocal
from app.models.assignment import Assignment, AssignmentQuestion, AssignmentUserTopup
from app.models.attempt import AssignmentAttempt, AssignmentAnswer
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Course, Module, Chapter
from app.models.course_extension import CourseExtensionJob
from app.models.user import User
from app.services import course_extension as svc


@pytest.fixture(autouse=True)
def _seed_users():
    with SessionLocal() as db:
        db.add(User(id=5, email="ext-svc-user-5@example.com", password_hash="x"))
        db.add(User(id=6, email="ext-svc-user-6@example.com", password_hash="x"))
        db.commit()


def _make_course():
    with SessionLocal() as db:
        course = Course(topic_slug="ext-svc-course", topic_raw="Ext Svc",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course


def test_bucket_is_created_once_and_reused():
    course = _make_course()
    with SessionLocal() as db:
        b1 = svc.get_or_create_bucket(db, 5, course)
        b2 = svc.get_or_create_bucket(db, 5, course)
        assert b1.id == b2.id
        assert b1.scope == "user" and b1.user_id == 5
        assert b1.title == "Additional Chapters"
        other = svc.get_or_create_bucket(db, 6, course)
        assert other.id != b1.id
        db.commit()


def test_append_chapters_adds_to_bucket_with_order():
    course = _make_course()
    with SessionLocal() as db:
        db.add(Module(course_id=course.id, title="G1", objective="o", order=1, scope="global"))
        db.flush()
        added = svc.append_chapters(db, 5, course, [
            {"title": "A", "objective": "oa"}, {"title": "B", "objective": "ob"},
        ])
        db.commit()
        assert len(added) == 2
        assert added[0]["title"] == "A"
        first = db.get(Chapter, added[0]["chapter_id"])
        assert first is not None
        bucket_id = first.module_id
        chapters = db.query(Chapter).filter_by(module_id=bucket_id).order_by(Chapter.order).all()
        assert [c.order for c in chapters] == [1, 2]
        db.commit()
    with SessionLocal() as db:
        added2 = svc.append_chapters(db, 5, course, [{"title": "C", "objective": "oc"}])
        db.commit()
        cid = added2[0]["chapter_id"]
        c = db.get(Chapter, cid)
        assert c is not None
        assert c.order == 3


def test_create_job_blocks_concurrent_run():
    course = _make_course()
    with SessionLocal() as db:
        svc.create_extension_job(db, 5, course, "first")
        db.commit()
        with pytest.raises(HTTPException) as exc:
            svc.create_extension_job(db, 5, course, "second")
        assert exc.value.status_code == 409
        svc.create_extension_job(db, 6, course, "other user okay")
        db.commit()


def test_get_job_and_list_and_delete():
    course = _make_course()
    with SessionLocal() as db:
        job = svc.create_extension_job(db, 5, course, "add networking")
        db.commit()
        fetched = svc.get_extension_job(db, 5, course, job.id)
        assert fetched.status == "pending"
        added = svc.append_chapters(db, 5, course, [{"title": "TCP", "objective": "o"}])
        db.commit()

        from app.models.course_extension import CourseExtensionJob as J
        job.status = "succeeded"
        job.result = added
        job.updated_at = datetime.now(timezone.utc)
        db.commit()

        listed = svc.list_extension_chapters(db, 5, course)
        assert [c["title"] for c in listed] == ["TCP"]
        assert listed[0]["content_ready"] is False

        svc.delete_extension_chapter(db, 5, course, added[0]["chapter_id"])
        db.commit()
        assert db.get(Chapter, added[0]["chapter_id"]) is None
        assert svc.list_extension_chapters(db, 5, course) == []
        assert db.query(Chapter).filter(Chapter.title == "TCP").count() == 0


def test_delete_rejects_foreign_or_global_chapters():
    course = _make_course()
    with SessionLocal() as db:
        m = Module(course_id=course.id, title="G", objective="o", order=1, scope="global")
        db.add(m)
        db.flush()
        g = Chapter(module_id=m.id, title="GlobalC", objective="o", order=1, scope="global")
        db.add(g)
        db.flush()
        with pytest.raises(HTTPException) as exc:
            svc.delete_extension_chapter(db, 5, course, g.id)
        assert exc.value.status_code == 404
        db.rollback()


def test_delete_cascades_content_assignments_attempts():
    course = _make_course()
    with SessionLocal() as db:
        bucket = svc.get_or_create_bucket(db, 5, course)
        db.flush()
        ch = Chapter(module_id=bucket.id, title="Del", objective="o", order=1, scope="user", user_id=5)
        db.add(ch)
        db.flush()
        content = ChapterContent(chapter_id=ch.id, version=1, scope="user", user_id=5,
                                 status="ready", outline=[], created_at=datetime.now(timezone.utc),
                                 updated_at=datetime.now(timezone.utc))
        db.add(content)
        db.flush()
        sec = ChapterContentSection(chapter_content_id=content.id, order=1, heading="H",
                                    kind="teaching", body_markdown="b", examples=[])
        db.add(sec)
        db.flush()
        assignment = Assignment(level="chapter", scope="user", user_id=5, chapter_content_id=content.id,
                                status="ready", created_at=datetime.now(timezone.utc),
                                updated_at=datetime.now(timezone.utc))
        db.add(assignment)
        db.flush()
        q = AssignmentQuestion(assignment_id=assignment.id, order=1, type="mcq", text="t",
                               correct_answer="a", explanation="e", concept_tag="c", difficulty="easy")
        db.add(q)
        db.flush()
        topup = AssignmentUserTopup(assignment_id=assignment.id, user_id=5, status="ready",
                                    created_at=datetime.now(timezone.utc),
                                    updated_at=datetime.now(timezone.utc))
        db.add(topup)
        db.flush()
        attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=5, status="graded",
                                    created_at=datetime.now(timezone.utc),
                                    updated_at=datetime.now(timezone.utc))
        db.add(attempt)
        db.commit()
        chapter_id, content_id = ch.id, content.id
        assignment_id, attempt_id = assignment.id, attempt.id
        svc.delete_extension_chapter(db, 5, course, chapter_id)
        db.commit()
        assert db.get(Chapter, chapter_id) is None
        assert db.get(ChapterContent, content_id) is None
        assert db.get(Assignment, assignment_id) is None
        assert db.get(AssignmentAttempt, attempt_id) is None
