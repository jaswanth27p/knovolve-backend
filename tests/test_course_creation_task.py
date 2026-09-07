from datetime import datetime, timezone
from unittest.mock import patch

from app.db import SessionLocal
from app.models.course import Course, CourseJob
from app.tasks.course_creation_task import run_course_creation_job


def _make_job(topic_slug="celery-test-topic"):
    with SessionLocal() as db:
        job = CourseJob(topic_slug=topic_slug, topic_raw="Celery Test Topic",
                          topic_embedding=[0.0] * 1024, status="pending",
                          created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def _make_course(topic_slug="celery-existing-course"):
    # CourseJob.course_id carries a real FK to courses.id, so a fake
    # final-state course id (e.g. a bare literal like 42) would violate the
    # constraint on commit. Insert a real row and use its actual id instead.
    with SessionLocal() as db:
        course = Course(topic_slug=topic_slug, topic_raw="Celery Existing Course",
                         topic_embedding=[0.0] * 1024,
                         created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course.id


def test_successful_run_marks_job_succeeded_with_course():
    job_id = _make_job()
    course_id = _make_course()
    fake_final_state = {"error": None, "existing_course_id": course_id}

    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        mock_build.return_value.invoke.return_value = fake_final_state
        run_course_creation_job(job_id)

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job.status == "succeeded"
        assert job.course_id == course_id


def test_exhausted_retries_marks_job_failed_no_partial_rows():
    job_id = _make_job(topic_slug="celery-fail-topic")

    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        mock_build.return_value.invoke.side_effect = RuntimeError("CourseGenerationError: exhausted retries")
        run_course_creation_job(job_id)

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job.status == "failed"
        assert job.error is not None
        assert job.course_id is None


def test_resume_uses_same_thread_id_as_job_id():
    job_id = _make_job(topic_slug="celery-resume-topic")
    course_id = _make_course(topic_slug="celery-resume-course")

    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        mock_build.return_value.invoke.return_value = {"error": None, "existing_course_id": course_id}
        run_course_creation_job(job_id)
        call_kwargs = mock_build.return_value.invoke.call_args.kwargs
        assert call_kwargs["config"]["configurable"]["thread_id"] == str(job_id)
