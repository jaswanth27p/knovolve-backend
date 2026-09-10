"""Celery task that runs one extension planning job for a CourseExtensionJob row.

Failures are recorded generically on the job (detail goes to server logs), and
persistence is all-or-nothing: a failed run writes zero chapters. Skeleton
generation is short, so no celery retry/checkpointer machinery is used — a
failed job stays 'failed' (the learner just retries from the page).
"""
import logging
from datetime import datetime, timezone

from celery import Task

from app.agents.course_extension.agent import plan_new_chapters
from app.db import SessionLocal
from app.models.course import Course
from app.models.course_extension import CourseExtensionJob
from app.services import course_extension
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

GENERIC_JOB_ERROR = "Extension failed. Please try again."


@celery_app.task(bind=True)
def run_course_extension_job(self: Task, job_id: int) -> None:
    with SessionLocal() as db:
        job = db.get(CourseExtensionJob, job_id)
        if job is None:
            raise ValueError(f"CourseExtensionJob {job_id} not found")
        if job.status in ("succeeded", "failed"):
            return
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
        try:
            course = db.get(Course, job.course_id)
            if course is None:
                raise ValueError(f"Course {job.course_id} not found")
            drafts = plan_new_chapters(db, course, job.user_id, job.request)
            # append_chapters creates the user's bucket unconditionally; a
            # planner that returns zero drafts must not leave an empty bucket
            # behind, so only append when there is something to append.
            if drafts:
                added = course_extension.append_chapters(db, job.user_id, course, drafts)
            else:
                added = []
            job.status = "succeeded"
            job.error = None
            job.result = added
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
        except Exception as exc:  # noqa: BLE001 - terminal, generic error surfaced
            db.rollback()
            logger.error("extension job %s failed", job_id, exc_info=exc)
            job.status = "failed"
            job.error = GENERIC_JOB_ERROR
            job.updated_at = datetime.now(timezone.utc)
            db.commit()