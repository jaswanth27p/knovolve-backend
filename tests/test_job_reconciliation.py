from datetime import datetime, timedelta, timezone

import pytest

from app.config import settings
from app.db import SessionLocal
from app.models.course import Course, CourseJob
from app.models.course_extension import CourseExtensionJob
from app.models.export import CourseGenerationRun, ExportJob
from app.models.user import User
from app.services.job_reconciliation import ORPHANED_JOB_ERROR, reconcile_stuck_jobs


@pytest.fixture(autouse=True)
def _seed_user():
    with SessionLocal() as db:
        db.add(User(id=9, email="reconcile-user-9@example.com", password_hash="x"))
        db.commit()


def _make_course(slug="reconcile-course"):
    with SessionLocal() as db:
        course = Course(topic_slug=slug, topic_raw="Reconcile Course",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course.id


def _stale_time(time_limit_seconds: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=time_limit_seconds) - timedelta(minutes=45)


def _fresh_time() -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=1)


def test_stale_running_course_job_marked_failed():
    with SessionLocal() as db:
        job = CourseJob(
            topic_slug="reconcile-job", topic_raw="Reconcile Job",
            topic_embedding=[0.0] * 2048, status="running",
            created_at=_stale_time(settings.celery_course_creation_time_limit_seconds),
            updated_at=_stale_time(settings.celery_course_creation_time_limit_seconds),
        )
        db.add(job)
        db.commit()
        job_id = job.id

    with SessionLocal() as db:
        updated = reconcile_stuck_jobs(db)
        assert updated >= 1
        db.expire_all()
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error == ORPHANED_JOB_ERROR


def test_fresh_running_course_job_left_alone():
    with SessionLocal() as db:
        job = CourseJob(
            topic_slug="reconcile-fresh-job", topic_raw="Reconcile Fresh Job",
            topic_embedding=[0.0] * 2048, status="running",
            created_at=_fresh_time(), updated_at=_fresh_time(),
        )
        db.add(job)
        db.commit()
        job_id = job.id

    with SessionLocal() as db:
        reconcile_stuck_jobs(db)
        db.expire_all()
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "running"


def test_stale_pending_extension_job_marked_failed():
    course_id = _make_course("reconcile-ext-course")
    with SessionLocal() as db:
        job = CourseExtensionJob(
            course_id=course_id, user_id=9, request="add more chapters", status="pending",
            created_at=_stale_time(settings.celery_task_time_limit_seconds),
            updated_at=_stale_time(settings.celery_task_time_limit_seconds),
        )
        db.add(job)
        db.commit()
        job_id = job.id

    with SessionLocal() as db:
        reconcile_stuck_jobs(db)
        db.expire_all()
        job = db.get(CourseExtensionJob, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error == ORPHANED_JOB_ERROR


def test_stale_running_export_job_marked_failed_with_completed_at():
    course_id = _make_course("reconcile-export-course")
    with SessionLocal() as db:
        job = ExportJob(
            user_id=9, course_id=course_id, kind="course", status="running",
            created_at=_stale_time(settings.celery_task_time_limit_seconds),
            updated_at=_stale_time(settings.celery_task_time_limit_seconds),
        )
        db.add(job)
        db.commit()
        job_id = job.id

    with SessionLocal() as db:
        reconcile_stuck_jobs(db)
        db.expire_all()
        job = db.get(ExportJob, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.completed_at is not None


def test_stale_running_generation_run_marked_failed():
    course_id = _make_course("reconcile-gen-course")
    with SessionLocal() as db:
        run = CourseGenerationRun(
            user_id=9, course_id=course_id, status="running",
            created_at=_stale_time(settings.celery_generation_run_time_limit_seconds),
            updated_at=_stale_time(settings.celery_generation_run_time_limit_seconds),
        )
        db.add(run)
        db.commit()
        run_id = run.id

    with SessionLocal() as db:
        reconcile_stuck_jobs(db)
        db.expire_all()
        run = db.get(CourseGenerationRun, run_id)
        assert run is not None
        assert run.status == "failed"


def test_succeeded_job_left_alone_even_if_old():
    with SessionLocal() as db:
        job = CourseJob(
            topic_slug="reconcile-succeeded-job", topic_raw="Reconcile Succeeded Job",
            topic_embedding=[0.0] * 2048, status="succeeded",
            created_at=_stale_time(settings.celery_course_creation_time_limit_seconds),
            updated_at=_stale_time(settings.celery_course_creation_time_limit_seconds),
        )
        db.add(job)
        db.commit()
        job_id = job.id

    with SessionLocal() as db:
        reconcile_stuck_jobs(db)
        db.expire_all()
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "succeeded"
