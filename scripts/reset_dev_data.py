"""Nuke dev data: course graph rows, checkpoint history, and the broker queue.

Development-only. We're pre-launch and keep no data worth preserving, so the
cleanest fix for schema/legacy-row quirks (raw errors in old jobs, zero-vector
embeddings in old rows) is to wipe them and start fresh rather than carry
workarounds in code.

Deletes:
- courses, modules, chapters, concepts, concept_edges, course_jobs
- langgraph checkpoints, checkpoint_blobs, checkpoint_writes
- the Redis keys Celery uses for broker/result (unacked/queued jobs die here,
  so no stale job can resurrect after a reset)

Keeps: users, refresh_tokens, alembic versions.

Usage: `python -m scripts.reset_dev_data`
"""

from __future__ import annotations

import redis
from sqlalchemy import text

from app.config import settings
from app.db import SessionLocal

_ORDERED_TABLES = (
    # child-first / FK-dependency order
    "concept_edges",
    "concepts",
    "chapters",
    "modules",
    "course_jobs",
    "courses",
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
)


def reset_dev_data() -> None:
    with SessionLocal() as db:
        db.execute(text("TRUNCATE %s RESTART IDENTITY CASCADE" % ", ".join(_ORDERED_TABLES)))
        db.commit()
        print("truncated:", ", ".join(_ORDERED_TABLES))

    r = redis.Redis.from_url(settings.redis_url)
    r.flushdb()
    print("flushed redis db at", settings.redis_url)


if __name__ == "__main__":
    reset_dev_data()