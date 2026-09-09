"""Chapter content service: resolve chapter routes and stream content events,
tailing pending diagram renders via redis pub/sub + DB reconcile."""
import json
import time
from typing import Iterator

import redis
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.chapter_content.generate import stream_chapter_content
from app.config import settings
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Course, Module, Chapter
from app.realtime.chapter_content_events import channel_name


def get_chapter(db: Session, course: Course, chapter_id: int) -> Chapter:
    chapter = (
        db.query(Chapter)
        .join(Module, Chapter.module_id == Module.id)
        .filter(Chapter.id == chapter_id, Module.course_id == course.id)
        .first()
    )
    if not chapter:
        raise HTTPException(status_code=404, detail="chapter not found")
    return chapter


def _reconcile_pending_diagrams(db: Session, chapter_content_id: int,
                                pending_orders: set[int]) -> list[dict]:
    """Resolve any pending diagram orders whose DB status has since moved to
    ready/failed (their publish may predate our subscribe). Returns the events
    to emit and mutates `pending_orders` to remove resolved orders."""
    events: list[dict] = []
    sections = db.scalars(
        select(ChapterContentSection).where(
            ChapterContentSection.chapter_content_id == chapter_content_id,
            ChapterContentSection.order.in_(pending_orders),
        )
    )
    for section in sections:
        if section.diagram_status == "ready":
            events.append({"type": "diagram_ready", "order": section.order,
                           "diagram_image_url": section.diagram_image_url})
            pending_orders.discard(section.order)
        elif section.diagram_status == "failed":
            events.append({"type": "diagram_failed", "order": section.order})
            pending_orders.discard(section.order)
    return events


def _tail_pending_diagrams(db: Session, chapter_content_id: int,
                           pending_orders: set[int]) -> Iterator[dict]:
    # Subscribe BEFORE draining/scanning. Redis pub/sub delivers only to live
    # subscribers, and the render task commits Postgres *before* it publishes, so:
    #   - anything committed before our SUBSCRIBE is invisible to the channel but
    #     caught by the periodic DB reconcile below;
    #   - anything committed after our SUBSCRIBE is delivered to us on the channel.
    # The old order (reconcile-then-subscribe) left a window — a diagram finishing
    # between the two was missed by both paths and stuck "pending" forever.
    client = redis.Redis.from_url(settings.redis_url)
    pubsub = client.pubsub()
    pubsub.subscribe(channel_name(chapter_content_id))
    deadline = time.monotonic() + settings.diagram_stream_timeout_seconds
    try:
        while pending_orders:
            resolved = _reconcile_pending_diagrams(db, chapter_content_id, pending_orders)
            for event in resolved:
                yield event
            if not pending_orders:
                break
            if time.monotonic() >= deadline:
                break
            message = pubsub.get_message(timeout=min(1.0, deadline - time.monotonic()))
            if message is None or message["type"] != "message":
                continue
            event = json.loads(message["data"])
            order = event.get("order")
            if order in pending_orders and event["type"] in ("diagram_ready", "diagram_failed"):
                yield event
                pending_orders.discard(order)
    finally:
        pubsub.close()

    # Final chance to reconcile (e.g. a diagram that resolved just as the deadline
    # hit and whose publish happened before our subscribe). Cheap, idempotent.
    for event in _reconcile_pending_diagrams(db, chapter_content_id, pending_orders):
        yield event


def stream_content_events(chapter: Chapter, db: Session, user_id: int) -> Iterator[dict]:
    pending_orders: set[int] = set()
    terminal: dict | None = None
    for event in stream_chapter_content(chapter, db, user_id):
        if event["type"] in ("done", "error"):
            # Hold the terminal event until after any pending-diagram tail so
            # the stream emits exactly one terminal ("done"|"error").
            terminal = event
            break
        yield event
        if event["type"] == "section_ready" and event["diagram_status"] == "pending":
            pending_orders.add(event["order"])

    if pending_orders and terminal and terminal["type"] == "done":
        # Only tail on a clean finish. If generation errored, the chapter is
        # failed and any earlier sections' diagrams are best resolved on the
        # next open's DB replay — blocking up to the timeout to hear about
        # them would only delay surfacing the error to the user.
        content = db.scalar(
            select(ChapterContent).where(ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global")
        )
        if content is not None:
            yield from _tail_pending_diagrams(db, content.id, pending_orders)

    yield terminal or {"type": "done"}
