from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.agents.course_creation.graph import CourseGenerationError
from app.db import SessionLocal
from app.models.course import Course, CourseJob
from app.tasks.course_creation_task import (
    GENERIC_JOB_ERROR,
    run_course_creation_job,
)


def _make_job(topic_slug="celery-test-topic", status="pending"):
    with SessionLocal() as db:
        job = CourseJob(topic_slug=topic_slug, topic_raw="Celery Test Topic",
                          topic_embedding=[0.0] * 1024, status=status,
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
    # Spacer row: conftest's clean_db TRUNCATEs course_jobs and courses together
    # with RESTART IDENTITY, so without this the first Course inserted after
    # the first CourseJob would coincidentally also land on id 1 — making
    # `job.course_id == course_id` pass even if the code under test mistakenly
    # assigned job.course_id = job_id instead of final_state["existing_course_id"].
    # Bumping the course sequence past job_id first keeps the ids distinguishable.
    _make_course(topic_slug="celery-spacer-course")
    course_id = _make_course()
    assert course_id != job_id
    fake_final_state = {"error": None, "existing_course_id": course_id}

    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        mock_build.return_value.invoke.return_value = fake_final_state
        # Celery tasks (bind=True) remain directly callable at runtime with just
        # the task's own args; the @celery_app.task decorator just makes pyright
        # infer a bound-method-style signature that appears to require "job_id"
        # as a second positional argument.
        run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None  # just inserted by _make_job above
        assert job.status == "succeeded"
        assert job.course_id == course_id
        assert job.error is None


def test_terminal_course_generation_error_marks_job_failed_generically():
    """Content that exhausted its repair budget (CourseGenerationError) is
    terminal: job -> failed. The persisted error must be the generic message,
    never the raw exception text (which can embed LLM output), and no partial
    course rows may remain."""
    job_id = _make_job(topic_slug="celery-fail-topic")
    raw_detail = "CourseGenerationError: exhausted retries on build_concept_graph after 2 attempts: ...llm content..."

    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        mock_build.return_value.invoke.side_effect = CourseGenerationError(raw_detail)
        run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error == GENERIC_JOB_ERROR
        assert raw_detail not in (job.error or "")
        assert job.course_id is None
        assert db.query(Course).filter_by(topic_slug="celery-fail-topic").count() == 0


def test_persist_failure_marks_job_failed():
    """A persist defect is a CourseGenerationError subclass -> terminal failed,
    never a false 'succeeded' with course_id=NULL."""
    job_id = _make_job(topic_slug="celery-persist-fail-topic")

    from app.agents.course_creation.graph import CoursePersistenceError
    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        mock_build.return_value.invoke.side_effect = CoursePersistenceError("persist_course failed: dup")
        run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.course_id is None


def test_transient_failure_propagates_for_retry_job_stays_running():
    """A transient infra error (anything not CourseGenerationError) must NOT be
    swallowed into a failed job. Under a direct call celery's self.retry()
    re-raises the original exception so the worker/runner can retry; the job
    row is left running (never failed) so the retry re-attempts it."""
    job_id = _make_job(topic_slug="celery-transient-topic")

    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        mock_build.return_value.invoke.side_effect = RuntimeError("db connection dropped")
        with pytest.raises(RuntimeError, match="db connection dropped"):
            run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "running"
        assert job.error is None


def test_already_succeeded_job_is_skipped():
    """A redelivered duplicate message must not rerun a job that already
    completed (idempotency guard)."""
    _make_course(topic_slug="celery-dup-spacer-course")
    course_id = _make_course(topic_slug="celery-done-course")
    job_id = _make_job(topic_slug="celery-done-topic", status="succeeded")
    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None
        job.course_id = course_id
        db.commit()

    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]
        mock_build.assert_not_called()

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.course_id == course_id


def test_resume_uses_same_thread_id_as_job_id():
    job_id = _make_job(topic_slug="celery-resume-topic")
    # See the spacer-row comment in test_successful_run_marks_job_succeeded_with_course:
    # keeps course_id from coincidentally equaling job_id after the shared
    # id-sequence reset, so the thread_id assertion below can't be satisfied
    # by accident.
    _make_course(topic_slug="celery-resume-spacer-course")
    course_id = _make_course(topic_slug="celery-resume-course")
    assert course_id != job_id

    with patch("app.tasks.course_creation_task.build_course_creation_graph") as mock_build:
        mock_build.return_value.invoke.return_value = {"error": None, "existing_course_id": course_id}
        run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]
        call_kwargs = mock_build.return_value.invoke.call_args.kwargs
        assert call_kwargs["config"]["configurable"]["thread_id"] == str(job_id)
