"""Celery beat task: backstop for orphaned generation/export jobs.

See app.services.job_reconciliation for why this exists -- per-task
soft/hard time limits handle a task that overruns while a worker is actively
running it, but not one purged from the queue or revoked before any worker
ever picked it up. This sweep is the last line of defense for both.
"""
import logging

from app.db import SessionLocal
from app.services.job_reconciliation import reconcile_stuck_jobs
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task
def reconcile_stuck_jobs_task() -> None:
    with SessionLocal() as db:
        updated = reconcile_stuck_jobs(db)
        if updated:
            logger.warning("reconciliation marked %s stuck job(s) as failed", updated)
