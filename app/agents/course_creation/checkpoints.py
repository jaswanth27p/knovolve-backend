"""LangGraph checkpoint lifecycle helpers.

The Postgres checkpointer stores every graph superstep as partial run state
(keyed by ``thread_id`` = ``str(job_id)``), so an interrupted run can resume
instead of regenerating. Those rows are durable but unbounded — every finished
job leaves one forever unless pruned. This module owns the two operations:

- :func:`purge_checkpoints` — drop all checkpoint rows for one thread. Used by
  the manual retry endpoint so a human-triggered retry regenerates fresh rather
  than blindly re-driving the failing node.
- :func:`prune_old_checkpoints` — delete checkpoints of finished jobs past the
  retention window, so the tables do not grow without bound.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from app.db import SessionLocal

logger = logging.getLogger(__name__)

#: child-first so removing a checkpoint never orphans pending writes/view data.
_CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


def purge_checkpoints(thread_id: str) -> None:
    """Delete every checkpoint row belonging to a thread."""
    with SessionLocal() as db:
        for table in _CHECKPOINT_TABLES:
            db.execute(
                text(f"DELETE FROM {table} WHERE thread_id = :tid"),
                {"tid": thread_id},
            )
        db.commit()
    logger.info("purged checkpoints for thread %s", thread_id)


def prune_old_checkpoints(retention_days: int) -> int:
    """Delete checkpoints of non-running jobs whose job rows are older than the
    retention window. Running jobs keep their checkpoints so crash-resume keeps
    working mid-run. Returns how many threads were pruned."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    expired_threads: list[str] = []
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT DISTINCT c.thread_id
                FROM checkpoints c
                JOIN course_jobs j ON j.id::text = c.thread_id
                WHERE j.status <> 'running'
                  AND j.updated_at < :cutoff
                """
            ),
            {"cutoff": cutoff},
        ).scalars().all()
        expired_threads = list(rows)
        for thread_id in expired_threads:
            for table in _CHECKPOINT_TABLES:
                db.execute(
                    text(f"DELETE FROM {table} WHERE thread_id = :tid"),
                    {"tid": thread_id},
                )
        db.commit()
    if expired_threads:
        logger.info(
            "pruned checkpoints for %d finished job(s) older than %d day(s)",
            len(expired_threads),
            retention_days,
        )
    return len(expired_threads)