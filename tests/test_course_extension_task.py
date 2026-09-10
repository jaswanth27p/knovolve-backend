from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.course_extension import CourseExtensionJob
from app.models.user import User
from app.tasks.course_extension_task import run_course_extension_job


@pytest.fixture(autouse=True)
def _seed_user():
    # The job rows and user-scoped bucket/module/chapter rows all FK
    # user_id -> users.id (see tests/test_course_extension_agent.py and
    # tests/test_course_extension_service.py for the same seed).
    with SessionLocal() as db:
        db.add(User(id=5, email="ext-task-user-5@example.com", password_hash="x"))
        db.commit()


def _make_course():
    with SessionLocal() as db:
        course = Course(topic_slug="ext-task-course", topic_raw="Ext Task",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course


def _make_job(course_id, user_id=5, status="pending"):
    with SessionLocal() as db:
        job = CourseExtensionJob(course_id=course_id, user_id=user_id, request="add tcp",
                                 status=status, created_at=datetime.now(timezone.utc),
                                 updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def test_success_appends_chapters_and_sets_result():
    course = _make_course()
    job_id = _make_job(course.id)
    drafts = [{"title": "TCP", "objective": "o"}, {"title": "UDP", "objective": "o"}]
    with patch("app.tasks.course_extension_task.plan_new_chapters", return_value=drafts) as mock_plan:
        run_course_extension_job(job_id)  # pyright: ignore[reportCallIssue]
    mock_plan.assert_called_once()
    with SessionLocal() as db:
        job = db.get(CourseExtensionJob, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.error is None
        assert job.result is not None
        assert len(job.result) == 2
        assert job.result[0]["title"] == "TCP"
        bucket = db.scalar(
            db.query(Module).filter_by(course_id=course.id, scope="user", user_id=5)
        )
        assert bucket is not None
        chapters = db.query(Chapter).filter_by(module_id=bucket.id).order_by(Chapter.order).all()
        assert [c.title for c in chapters] == ["TCP", "UDP"]


def test_zero_chapters_succeeds_without_bucket():
    course = _make_course()
    job_id = _make_job(course.id)
    with patch("app.tasks.course_extension_task.plan_new_chapters", return_value=[]):
        run_course_extension_job(job_id)  # pyright: ignore[reportCallIssue]
    with SessionLocal() as db:
        job = db.get(CourseExtensionJob, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.result == []
        assert db.query(Module).filter_by(course_id=course.id, scope="user", user_id=5).count() == 0


def test_failure_marks_job_failed_and_writes_nothing():
    course = _make_course()
    job_id = _make_job(course.id)
    with patch("app.tasks.course_extension_task.plan_new_chapters",
               side_effect=ValueError("budget exceeded")):
        run_course_extension_job(job_id)  # pyright: ignore[reportCallIssue]
    with SessionLocal() as db:
        job = db.get(CourseExtensionJob, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error is not None
        assert "budget exceeded" not in (job.error or "")  # generic only
        assert db.query(Module).filter_by(course_id=course.id, scope="user", user_id=5).count() == 0


def test_already_succeeded_job_skipped():
    course = _make_course()
    job_id = _make_job(course.id, status="succeeded")
    with patch("app.tasks.course_extension_task.plan_new_chapters") as mock_plan:
        run_course_extension_job(job_id)  # pyright: ignore[reportCallIssue]
        mock_plan.assert_not_called()