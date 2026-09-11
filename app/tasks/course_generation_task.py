"""Celery worker that completes one user-scoped full-course generation run."""
import logging
from datetime import datetime, timezone

from celery import Task

from app.agents.generation.ensure import (
    ensure_chapter_assignment,
    ensure_chapter_content,
    ensure_module_assignment,
    ensure_remediation_content,
)
from app.db import SessionLocal
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter
from app.models.export import CourseGenerationRun
from app.services import generation as generation_service
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

GENERIC_GENERATION_ERROR = "Course generation failed. Please try again."


def _execute_unit(db, run, unit) -> None:
    kind = unit["kind"]
    if kind == "content":
        chapter = db.get(Chapter, unit["chapter_id"])
        if chapter is None:
            raise ValueError(f"Chapter {unit['chapter_id']} not found")
        ensure_chapter_content(db, chapter)
    elif kind == "chapter_assignment":
        content = db.get(ChapterContent, unit["content_id"])
        if content is None:
            raise ValueError(f"ChapterContent {unit['content_id']} not found")
        ensure_chapter_assignment(db, content)
    elif kind == "remediation_content":
        content = db.get(ChapterContent, unit["content_id"])
        if content is None:
            raise ValueError(f"ChapterContent {unit['content_id']} not found")
        ensure_remediation_content(db, unit["chapter_id"], run.user_id, content)
    elif kind == "remediation_assignment":
        content = db.get(ChapterContent, unit["content_id"])
        if content is None:
            raise ValueError(f"ChapterContent {unit['content_id']} not found")
        ensure_chapter_assignment(db, content)
    elif kind == "module_assignment":
        ensure_module_assignment(db, unit["module_id"])
    else:
        raise ValueError(f"Unsupported generation unit: {kind}")


@celery_app.task(bind=True)
def run_course_generation_task(self: Task, run_id: int) -> None:
    with SessionLocal() as db:
        run = db.get(CourseGenerationRun, run_id)
        if run is None:
            raise ValueError(f"CourseGenerationRun {run_id} not found")
        if run.status in ("succeeded", "failed"):
            return
        run.status = "running"
        run.updated_at = datetime.now(timezone.utc)
        db.commit()
        failed = False
        try:
            for unit in list(run.unit_states or []):
                generation_service.advance_run(db, run, unit["unit_id"], "running")
                try:
                    _execute_unit(db, run, unit)
                except Exception as exc:
                    logger.error("generation run %s unit %s failed", run_id, unit["unit_id"], exc_info=exc)
                    generation_service.advance_run(db, run, unit["unit_id"], "failed")
                    failed = True
                else:
                    generation_service.advance_run(db, run, unit["unit_id"], "done")
            run.status = "failed" if failed else "succeeded"
            run.error = GENERIC_GENERATION_ERROR if failed else None
            run.completed_at = datetime.now(timezone.utc)
            run.updated_at = datetime.now(timezone.utc)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error("generation run %s failed", run_id, exc_info=exc)
            run.status = "failed"
            run.error = GENERIC_GENERATION_ERROR
            run.completed_at = datetime.now(timezone.utc)
            run.updated_at = datetime.now(timezone.utc)
            db.commit()
