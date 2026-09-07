"""Celery task that renders one section's diagram and uploads it to MinIO.

Decoupled from chapter-content text generation (Task 8) on purpose: a
diagram is optional polish, never a blocker for the section's teaching
text. Bounded retry (max_retries=3, exponential backoff) means a transient
render/upload failure gets a few chances, but a persistently broken spec
gives up permanently (`diagram_status="failed"`) instead of retrying
forever — the frontend shows a "diagram unavailable" fallback in that case.
"""

import logging
from celery import Task
from celery.exceptions import MaxRetriesExceededError
from app.db import SessionLocal
from app.diagrams.render import render_diagram_svg
from app.models.chapter_content import ChapterContentSection
from app.realtime.chapter_content_events import publish_event
from app.storage.s3 import ensure_bucket, get_public_url, upload_object
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

RETRY_BASE_DELAY_SECONDS = 5
MAX_RETRY_DELAY_SECONDS = 30
MAX_DIAGRAM_RETRIES = 3


def _retry_delay_seconds(retries_so_far: int) -> int:
    return min(MAX_RETRY_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2 ** retries_so_far))


def render_and_upload_diagram(section: ChapterContentSection) -> str:
    """Render `section.diagram_spec` and upload it, returning the public
    URL. Raises on failure — the caller decides retry vs. give-up policy."""
    svg_bytes = render_diagram_svg(section.diagram_spec)
    ensure_bucket()
    key = f"images/chapter-content-sections/{section.id}.svg"
    upload_object(key, svg_bytes, "image/svg+xml")
    return get_public_url(key)


@celery_app.task(bind=True, max_retries=MAX_DIAGRAM_RETRIES, default_retry_delay=RETRY_BASE_DELAY_SECONDS)
def render_diagram_task(self: Task, section_id: int) -> None:
    with SessionLocal() as db:
        section = db.get(ChapterContentSection, section_id)
        if section is None:
            raise ValueError(f"ChapterContentSection {section_id} not found")
        if section.diagram_status != "pending":
            return  # already resolved (duplicate delivery) or no diagram

        try:
            url = render_and_upload_diagram(section)
        except Exception as exc:
            section.diagram_attempts += 1
            db.commit()
            try:
                self.retry(exc=exc, countdown=_retry_delay_seconds(self.request.retries))
            except MaxRetriesExceededError:
                retries_exhausted = True
            except Exception:
                # Celery's Task.retry() (5.6.3): when `exc=` is passed and the
                # retry budget is exhausted, it re-raises the ORIGINAL
                # exception via raise_with_context(exc) instead of raising
                # MaxRetriesExceededError (that branch only fires when no
                # `exc` was given) — see celery/app/task.py:Task.retry. Since
                # we must pass exc= to get the real failure re-raised under a
                # direct call (no worker/broker), detect exhaustion by retry
                # count instead of relying on the exception type.
                if self.request.retries < self.max_retries:
                    raise
                retries_exhausted = True
            else:
                retries_exhausted = False  # pragma: no cover - retry() never returns when throw=True

            if retries_exhausted:
                section.diagram_status = "failed"
                db.commit()
                publish_event(section.chapter_content_id,
                              {"type": "diagram_failed", "order": section.order})
            return

        section.diagram_status = "ready"
        section.diagram_image_url = url
        db.commit()
        publish_event(section.chapter_content_id,
                      {"type": "diagram_ready", "order": section.order, "diagram_image_url": url})
