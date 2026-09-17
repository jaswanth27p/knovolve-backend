"""Celery wrappers for assignment generation.

Retry taxonomy (mirrors ``course_creation_task.py``): the generators
(``app.agents.assignment.generate``) catch LLM/persistence errors in their own
body and mark the Assignment/AssignmentUserTopup row ``failed`` — a terminal
outcome this wrapper must NOT retry (re-running would regenerate the same
content). Those errors therefore never reach here. What *can* escape is a
setup/claim failure before that try block — most importantly a DB
``OperationalError`` during get-or-create or the retryable-claim CAS — which is
transient infra: hand it to ``Task.retry`` with exponential backoff so the
worker re-attempts it. A ``ValueError`` (missing/deleted row) is a caller bug,
not transient, so it propagates immediately without consuming the budget.
"""

import logging

from celery import Task

from app.db import SessionLocal
from app.agents.assignment.generate import (
    generate_chapter_assignment,
    generate_module_assignment,
    generate_module_topup,
)
from app.llm.langfuse_client import flush_langfuse
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

RETRY_BASE_DELAY_SECONDS = 5
MAX_RETRY_DELAY_SECONDS = 60
MAX_CELERY_RETRIES = 3


def _transient_retry_delay_seconds(task: Task) -> int:
    """Exponential backoff for celery retry countdown, doubling per attempt."""
    return min(
        MAX_RETRY_DELAY_SECONDS,
        RETRY_BASE_DELAY_SECONDS * (2 ** task.request.retries),
    )


def _retry_if_transient(task: Task, exc: Exception) -> None:
    """Re-raise a caller bug; otherwise hand a transient error to Task.retry."""
    if isinstance(exc, ValueError):
        # Missing/deleted row or similar invariant: another attempt cannot fix
        # it, so fail fast instead of burning the retry budget.
        raise exc
    logger.warning(
        "assignment task hit transient failure, retry #%s",
        task.request.retries + 1,
        exc_info=exc,
    )
    task.retry(
        exc=exc,
        countdown=_transient_retry_delay_seconds(task),
        max_retries=MAX_CELERY_RETRIES,
    )


@celery_app.task(bind=True, max_retries=MAX_CELERY_RETRIES,
                 default_retry_delay=RETRY_BASE_DELAY_SECONDS)
def generate_chapter_assignment_task(self: Task, chapter_content_id: int) -> None:
    with SessionLocal() as db:
        try:
            generate_chapter_assignment(chapter_content_id, db)
        except Exception as exc:  # noqa: BLE001 - taxonomy above
            _retry_if_transient(self, exc)
        finally:
            flush_langfuse()


@celery_app.task(bind=True, max_retries=MAX_CELERY_RETRIES,
                 default_retry_delay=RETRY_BASE_DELAY_SECONDS)
def generate_module_assignment_task(self: Task, module_id: int) -> None:
    with SessionLocal() as db:
        try:
            generate_module_assignment(module_id, db)
        except Exception as exc:  # noqa: BLE001 - taxonomy above
            _retry_if_transient(self, exc)
        finally:
            flush_langfuse()


@celery_app.task(bind=True, max_retries=MAX_CELERY_RETRIES,
                 default_retry_delay=RETRY_BASE_DELAY_SECONDS)
def generate_module_topup_task(self: Task, assignment_id: int, user_id: int) -> None:
    with SessionLocal() as db:
        try:
            generate_module_topup(assignment_id, user_id, db)
        except Exception as exc:  # noqa: BLE001 - taxonomy above
            _retry_if_transient(self, exc)
        finally:
            flush_langfuse()
