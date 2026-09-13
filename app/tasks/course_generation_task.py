"""Celery worker that completes one user-scoped full-course generation run.

Units are independent across chapters, so they run concurrently instead of one
at a time. Concurrency is bounded twice over: within a run by
``generation_run_max_parallel_units`` worker threads (each with its own DB
session), and globally by the Celery worker's ``concurrency`` child processes.
Each unit's own chapter content also fans its sections out (see
``app.agents.generation.ensure``), so the true LLM concurrency ceiling is
roughly ``worker_concurrency * generation_run_max_parallel_units *
chapter_section_max_workers`` -- tune those three down to shrink memory.

Dependencies are expressed per unit as ``depends_on`` (built at plan time in
``app.services.generation``). Units are grouped into waves by longest
dependency chain, and a wave completes before the next starts, so a chapter's
assignment always runs after its content and a module assignment after its
chapters' content.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from celery import Task

from app.agents.generation.ensure import (
    ensure_chapter_assignment,
    ensure_chapter_content,
    ensure_module_assignment,
    ensure_remediation_content,
)
from app.config import settings
from app.db import SessionLocal
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter
from app.models.export import CourseGenerationRun
from app.services import exports as export_service
from app.services import generation as generation_service
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

GENERIC_GENERATION_ERROR = "Course generation failed. Please try again."


def _execute_unit(db, user_id: int, unit) -> None:
    kind = unit["kind"]
    if kind == "content":
        chapter = db.get(Chapter, unit["chapter_id"])
        if chapter is None:
            raise ValueError(f"Chapter {unit['chapter_id']} not found")
        ensure_chapter_content(db, chapter)
    elif kind == "chapter_assignment":
        # Content id is only known up front when the content already existed at
        # plan time; when this run created it, resolve the (now persisted) base
        # content from the chapter.
        if unit.get("content_id") is not None:
            content = db.get(ChapterContent, unit["content_id"])
        else:
            chapter = db.get(Chapter, unit["chapter_id"])
            if chapter is None:
                raise ValueError(f"Chapter {unit['chapter_id']} not found")
            content = export_service.chapter_base_content(db, chapter, user_id)
        if content is None:
            raise ValueError(f"Chapter {unit['chapter_id']} has no base content for its assignment")
        ensure_chapter_assignment(db, content)
    elif kind == "remediation_content":
        content = db.get(ChapterContent, unit["content_id"])
        if content is None:
            raise ValueError(f"ChapterContent {unit['content_id']} not found")
        ensure_remediation_content(db, unit["chapter_id"], user_id, content)
    elif kind == "remediation_assignment":
        content = db.get(ChapterContent, unit["content_id"])
        if content is None:
            raise ValueError(f"ChapterContent {unit['content_id']} not found")
        ensure_chapter_assignment(db, content)
    elif kind == "module_assignment":
        ensure_module_assignment(db, unit["module_id"])
    else:
        raise ValueError(f"Unsupported generation unit: {kind}")


def _dependency_levels(units: list[dict]) -> list[list[dict]]:
    """Group units into waves by longest dependency chain.

    ``depends_on`` only ever references units that appear earlier in the same
    plan, so a single forward pass resolves every level. Units with no unmet
    dependency (level 0) run together; their dependents form the next wave.
    """
    level: dict[str, int] = {}
    for unit in units:
        deps = unit.get("depends_on") or []
        level[unit["unit_id"]] = 0 if not deps else 1 + max(
            level.get(dep, 0) for dep in deps
        )
    waves: dict[int, list[dict]] = {}
    for unit in units:
        waves.setdefault(level[unit["unit_id"]], []).append(unit)
    return [waves[key] for key in sorted(waves)]


def _run_unit_isolated(run_id: int, user_id: int, unit: dict) -> tuple[str, str]:
    """Run one unit in its own thread and DB session.

    The HTTP/LLM work inside releases the GIL, so threads give real
    concurrency; the session is per-thread because SQLAlchemy sessions are not
    thread-safe. Returns ``(unit_id, "done"|"failed")`` — the caller owns all
    writes to the run row, so concurrent units never race on its JSONB state.
    """
    with SessionLocal() as db:
        try:
            _execute_unit(db, user_id, unit)
        except Exception as exc:
            logger.error(
                "generation run %s unit %s failed", run_id, unit["unit_id"], exc_info=exc
            )
            return unit["unit_id"], "failed"
    return unit["unit_id"], "done"


@celery_app.task(
    bind=True,
    time_limit=settings.celery_generation_run_time_limit_seconds,
    soft_time_limit=settings.celery_generation_run_time_limit_seconds,
)
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
            units = list(run.unit_states or [])
            max_workers = max(1, settings.generation_run_max_parallel_units)
            for wave in _dependency_levels(units):
                with ThreadPoolExecutor(max_workers=min(max_workers, len(wave))) as executor:
                    futures = [
                        executor.submit(_run_unit_isolated, run_id, run.user_id, unit)
                        for unit in wave
                    ]
                    for future in as_completed(futures):
                        unit_id, status = future.result()
                        generation_service.advance_run(db, run, unit_id, status)
                        if status == "failed":
                            failed = True
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
