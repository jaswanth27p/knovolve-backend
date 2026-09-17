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
from langchain_core.runnables import RunnableConfig

from app.agents.course_creation.checkpoints import prune_old_checkpoints
from app.agents.course_creation.graph import (
    CourseGenerationError,
    build_course_creation_graph,
)
from app.agents.course_creation.state import CourseCreationState
from app.config import settings
from app.db import SessionLocal
from app.llm.langfuse_client import flush_langfuse, traced_workflow
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


def _has_checkpoint(state) -> bool:
    """Whether a snapshot came from a persisted checkpoint.

    langgraph's ``get_state`` returns a default snapshot with ``metadata=None``
    (and no ``checkpoint_id`` in its config) when the thread has no checkpoint
    at all; a real checkpoint always carries non-None metadata (verified against
    langgraph 1.2.11). Treating the no-checkpoint case as "completed" would
    return an empty state and blank-page the job, so it must be distinguished
    from a genuinely finished run.
    """
    if state is None:
        return False
    if getattr(state, "metadata", None) is not None:
        return True
    configurable = (getattr(state, "config", None) or {}).get("configurable") or {}
    return bool(configurable.get("checkpoint_id"))


def _invoke_generation(graph, initial_state: CourseCreationState, config: RunnableConfig,
                       job_id: int) -> dict:
    """Run the graph, or return an already-finished run's saved state.

    Every invoke uses synchronous durability so every completed superstep is
    checkpointed before the next runs — that is what makes crash-resume
    possible (a worker killed mid-run leaves a partially-written run whose
    ``next`` node is pending).

    Three cases, distinguished from the snapshot:

    - No checkpoint at all (fresh thread): invoke from ``initial_state``.
    - Checkpoint with a non-empty ``next`` (interrupted): re-invoke with
      ``None`` input so langgraph re-drives only the still-pending node from
      its checkpoint instead of regenerating the whole course.
    - A checkpoint with empty ``next`` is COMPLETED. A worker crash between
      graph completion and the ``job.status='succeeded'`` commit leaves the
      job "running", so the redelivered task must return the checkpointed
      values instead of re-running the whole pipeline.
    """
    try:
        state = graph.get_state(config)
    except Exception:  # noqa: BLE001 - a broken read must not block generation
        state = None
    if state is not None and _has_checkpoint(state):
        if state.next:
            logger.info(
                "job %s: resuming interrupted run at %s instead of regenerating",
                job_id,
                list(state.next),
            )
            return graph.invoke(None, config=config, durability="sync")
        logger.info(
            "job %s: checkpoint already completed; returning saved state", job_id
        )
        return state.values
    return graph.invoke(initial_state, config=config, durability="sync")


def _finalize_failed(job_id: int, exc: BaseException) -> None:
    """Log full detail; persist only a generic message on the job row."""
    logger.error(
        "course job %s failed terminally (status remains observable as failed)",
        job_id,
        exc_info=exc,
    )
    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        if job is None:
            return
        job.status = "failed"
        job.error = GENERIC_JOB_ERROR
        job.updated_at = datetime.now(timezone.utc)
        db.commit()


@celery_app.task(
    bind=True, max_retries=MAX_CELERY_RETRIES, default_retry_delay=5,
    time_limit=settings.celery_course_creation_time_limit_seconds,
    soft_time_limit=settings.celery_course_creation_soft_time_limit_seconds,
)
def run_course_creation_job(self: Task, job_id: int) -> None:
    # Each DB session below is scoped narrowly (open, write, close) rather
    # than held open across the graph invocation. The graph can run for many
    # minutes on LLM/web calls; a session left open that whole time is at
    # best a wasted connection and at worst a self-deadlock — it previously
    # did exactly that against the checkpointer's one-time schema setup (see
    # app/agents/course_creation/checkpointer.py).
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

        # Opportunistic checkpoint hygiene: finished jobs' partial state rows
        # age out, so the checkpoints tables don't grow forever. Failure to
        # prune must never fail the job itself.
        try:
            prune_old_checkpoints(settings.checkpoint_retention_days)
        except Exception:  # noqa: BLE001
            logger.warning(
                "checkpoint pruning failed for job %s", job_id, exc_info=True
            )

        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)

        # Read every job field the graph needs before commit: SQLAlchemy
        # expires ORM attributes on commit by default, and touching them
        # afterward would silently re-open a transaction on this same session
        # — which previously stayed open (unnoticed) for the graph's whole
        # multi-minute run.
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
        if job.allow_duplicate:
            # Force-created job: tell normalize_topic to skip semantic dedup so
            # the requested course is built instead of re-merged.
            initial_state["allow_duplicate"] = True

        created_by_user_id = job.created_by_user_id
        db.commit()

    try:
        with traced_workflow(
            "Course Creation", user_id=created_by_user_id, session_id=job_id, tags=["course-creation"],
        ):
            graph = build_course_creation_graph()
            final_state = _invoke_generation(
                graph,
                initial_state,
                {"configurable": {"thread_id": str(job_id)}},
                job_id,
            )
    except CourseGenerationError as exc:
        # Content that exhausted its in-graph repair budget, or a persist
        # defect. Regenerating would reproduce the same failure.
        _finalize_failed(job_id, exc)
        return
    except Exception as exc:
        # Transient infra error that escaped the tenacity layer -- this also
        # catches celery.exceptions.SoftTimeLimitExceeded (a plain Exception
        # subclass), raised in-process just before the hard time_limit kill
        # configured on this task's decorator. Retrying a timeout is safe
        # specifically because this graph is checkpointed: the next attempt
        # resumes via _invoke_generation instead of starting over. Under a
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
            _finalize_failed(job_id, exc)
        return
    finally:
        flush_langfuse()

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        if job is None:
            raise ValueError(f"CourseJob {job_id} not found")
        job.status = "succeeded"
        job.course_id = final_state["existing_course_id"]
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
