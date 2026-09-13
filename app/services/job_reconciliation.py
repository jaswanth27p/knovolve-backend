"""Periodic backstop that marks orphaned generation/export jobs as failed.

Per-task soft/hard Celery time limits (see celery_app.py) catch a task that is
still running past its budget. They cannot catch a task purged from the queue,
revoked before any worker picked it up, or lost to a worker crash that slips
past task_reject_on_worker_lost -- in all of those cases no task code ever
runs, so no `except` block ever fires to flag the row. Without this sweep such
a row stays "pending"/"running" forever and the frontend polls a job that will
never resolve.

STALE_AFTER per model mirrors app.agents.assignment.generate.STALE_AFTER's
reasoning: it must exceed the longest a *healthy* run can hold the row (that
job type's own time_limit), plus a margin for clock skew between the worker
that last wrote updated_at and this sweep.
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.course import CourseJob
from app.models.course_extension import CourseExtensionJob
from app.models.export import CourseGenerationRun, ExportJob

ORPHANED_JOB_ERROR = (
    "Job did not finish -- it may have been interrupted or cleared from the "
    "queue. Please retry."
)

_MARGIN = timedelta(minutes=30)

# Each entry's stale-after window is that job type's own worst-case time_limit
# plus _MARGIN. Assignment/AssignmentUserTopup rows are deliberately excluded:
# they already self-heal via a reclaim-on-read CAS (see
# app.agents.assignment.generate._reactivate_if_retryable), so sweeping them
# here would just duplicate that mechanism.
_STALE_AFTER_SECONDS_BY_MODEL = {
    CourseJob: settings.celery_course_creation_time_limit_seconds,
    CourseExtensionJob: settings.celery_task_time_limit_seconds,
    ExportJob: settings.celery_task_time_limit_seconds,
    CourseGenerationRun: settings.celery_generation_run_time_limit_seconds,
}


def reconcile_stuck_jobs(db: Session) -> int:
    """Flip any pending/running job past its stale threshold to failed.

    Returns the number of rows updated."""
    now = datetime.now(timezone.utc)
    updated = 0
    for model, time_limit_seconds in _STALE_AFTER_SECONDS_BY_MODEL.items():
        cutoff = now - timedelta(seconds=time_limit_seconds) - _MARGIN
        rows = db.scalars(
            select(model).where(
                model.status.in_(["pending", "running"]),
                model.updated_at < cutoff,
            )
        ).all()
        for row in rows:
            row.status = "failed"
            row.error = ORPHANED_JOB_ERROR
            row.updated_at = now
            if hasattr(row, "completed_at"):
                row.completed_at = now
            updated += 1
        if rows:
            db.commit()
    return updated
