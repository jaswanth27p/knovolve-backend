"""Celery wrapper for grading one AssignmentAttempt.

Deliberately has NO Celery retry/backoff, unlike ``assignment_tasks.py``:
``app.agents.evaluation.grade.grade_assignment_attempt`` catches every
exception from its grading body, rolls back, and marks the attempt ``failed``
before returning — it never raises, so ``Task.retry`` would never fire and a
retry policy here cannot change outcomes. Making it retryable would require
that function to re-raise transient infra errors (DB/LLM timeouts) while
keeping terminal grading failures terminal; that is out of scope for this
module and would alter grading semantics. The task is safely idempotent on
redelivery anyway: grading no-ops on an attempt whose status is no longer
"grading".
"""
from app.db import SessionLocal
from app.agents.evaluation.grade import grade_assignment_attempt
from app.llm.langfuse_client import flush_langfuse
from app.tasks.celery_app import celery_app


@celery_app.task
def grade_assignment_attempt_task(attempt_id: int) -> None:
    try:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)
    finally:
        flush_langfuse()
