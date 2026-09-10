from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.course_extension import CourseExtensionJob
from app.models.user import User


@pytest.fixture(autouse=True)
def _seed_default_user():
    with SessionLocal() as db:
        db.add(User(id=7, email="ext-model-user@example.com", password_hash="x"))
        db.commit()


def _make_course():
    with SessionLocal() as db:
        course = Course(topic_slug="ext-model-course", topic_raw="Ext Model",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course.id


def _make_job(course_id, user_id, status="pending"):
    with SessionLocal() as db:
        job = CourseExtensionJob(course_id=course_id, user_id=user_id, request="add xyz",
                                 status=status, created_at=datetime.now(timezone.utc),
                                 updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def test_modules_and_chapters_default_to_global_scope():
    course_id = _make_course()
    with SessionLocal() as db:
        m = Module(course_id=course_id, title="M", objective="o", order=1)
        db.add(m)
        db.commit()
        db.refresh(m)
        assert m.scope == "global"
        assert m.user_id is None
        c = Chapter(module_id=m.id, title="C", objective="o", order=1)
        db.add(c)
        db.commit()
        db.refresh(c)
        assert c.scope == "global"
        assert c.user_id is None


def test_one_active_extension_job_per_user_course():
    course_id = _make_course()
    job_id = _make_job(course_id, user_id=7)
    with SessionLocal() as db:
        job = db.get(CourseExtensionJob, job_id)
        assert job is not None
        job.status = "succeeded"
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
    second_run_id = _make_job(course_id, user_id=7)  # a second run after success is fine
    with SessionLocal() as db:
        job = db.get(CourseExtensionJob, second_run_id)
        assert job is not None
        job.status = "succeeded"
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
    # an active (pending/running) run blocks a second one
    _make_job(course_id, user_id=7, status="running")
    try:
        _make_job(course_id, user_id=7, status="pending")
        raise AssertionError("expected IntegrityError on second active run")
    except IntegrityError:
        pass
