"""Celery task that runs the course-creation graph for one CourseJob row.

Failure-handling contract (see the full taxonomy in the design doc):

- CourseGenerationError / CoursePersistenceError (content that could not be
  repaired within the graph's own retry budget, or a persistence defect) are
  TERMINAL: mark the job failed immediately with a generic error and log the
  detail. Retrying would regenerate the same broken content.
- CourseJob-not-found and similar invariants are programmer bugs: raise so the
  worker records a hard failure.
- Everything else (LLM/embedding timeouts that escaped tenacity, DB
  OperationalError, transient broker/checkpointer failures) is TRANSIENT:
  self.retry() with exponential backoff. A rerun re-enters at normalize_topic,
  which short-circuits cheaply if a course for the topic now exists (the
  duplicate-publish race) and otherwise regenerates. Once the celery retry
  budget is exhausted the job is marked failed.
- acks_late + visibility_timeout (celery_app.py) turn a worker crash mid-run
  into a broker redelivery rather than a permanently-stuck "running" job.

Job.error only ever holds a generic message; full exception detail goes to the
server logs so authenticated API callers can never read raw provider/DB text.
"""

import logging
from datetime import datetime, timezone

from celery import Task
from celery.exceptions import MaxRetriesExceededError

from app.agents.course_creation.graph import (
    CourseGenerationError,
    build_course_creation_graph,
)
from app.agents.course_creation.state import CourseCreationState
from app.db import SessionLocal
from app.models.course import CourseJob
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

#: User-facing, logged-out-of-detail failure message written to job.error.
GENERIC_JOB_ERROR = "Course generation failed. Please try again or retry this job."

#: Terminal retries never enter the graph again; transient retries back off
#: exponentially (5s, 10s, 20s, ... capped at MAX_TRANSIENT_RETRY_DELAY).
RETRY_BASE_DELAY_SECONDS = 5
MAX_RETRY_DELAY_SECONDS = 60

MAX_CELERY_RETRIES = 3


def _transient_retry_delay_seconds(task: Task) -> int:
    """Exponential backoff for celery retry countdown, doubling per attempt."""
    return min(
        MAX_RETRY_DELAY_SECONDS,
        RETRY_BASE_DELAY_SECONDS * (2 ** task.request.retries),
    )


def _finalize_failed(db, job: CourseJob, exc: BaseException) -> None:
    """Log full detail; persist only a generic message on the job row."""
    logger.error(
        "course job %s failed terminally (status remains observable as failed)",
        job.id,
        exc_info=exc,
    )
    job.status = "failed"
    job.error = GENERIC_JOB_ERROR
    job.updated_at = datetime.now(timezone.utc)
    db.commit()


@celery_app.task(bind=True, max_retries=MAX_CELERY_RETRIES, default_retry_delay=5)
def run_course_creation_job(self: Task, job_id: int) -> None:
    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        if job is None:
            # `assert` is stripped under -O/PYTHONOPTIMIZE, so a missing row
            # (a genuine caller bug — Task 11 only enqueues ids it just
            # inserted) must be an explicit, non-optimizable failure rather
            # than a debug-mode-only invariant.
            raise ValueError(f"CourseJob {job_id} not found")

        # A redelivered duplicate message (worker died after a previous attempt
        # already committed a terminal state, before the ack) must not rerun a
        # finished job. Failed jobs are only resurrected deliberately, via the
        # manual retry endpoint, which flips them back to pending.
        if job.status in ("succeeded", "failed"):
            return

        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        db.commit()

        initial_state: CourseCreationState = {
            "job_id": job_id,
            "topic_raw": job.topic_raw,
            "topic_slug": job.topic_slug,
            "topic_embedding": job.topic_embedding,
            "existing_course_id": None,
            "modules": None,
            "concepts": None,
            "concept_edges": None,
            "error": None,
        }

        try:
            graph = build_course_creation_graph()
            final_state = graph.invoke(
                initial_state, config={"configurable": {"thread_id": str(job_id)}}
            )
        except CourseGenerationError as exc:
            # Content that exhausted its in-graph repair budget, or a persist
            # defect. Regenerating would reproduce the same failure.
            _finalize_failed(db, job, exc)
            return
        except Exception as exc:
            # Transient infra error that escaped the tenacity layer. Under a
            # direct call (tests / no broker) retry() re-raises `exc`; under a
            # worker it re-queues and raises Retry. Either way control does not
            # fall through to marking the job succeeded.
            logger.warning(
                "course job %s hit transient failure, retry #%s",
                job_id,
                self.request.retries + 1,
                exc_info=True,
            )
            try:
                self.retry(
                    exc=exc,
                    countdown=_transient_retry_delay_seconds(self),
                    max_retries=MAX_CELERY_RETRIES,
                )
            except MaxRetriesExceededError:
                _finalize_failed(db, job, exc)
            return

        job.status = "succeeded"
        job.course_id = final_state["existing_course_id"]
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
