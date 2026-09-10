"""Course preview: similarity candidates + the Redis-backed canonicalization cache.

API flow (see create_course_job):
  1. First POST /courses canonicalizes+embeds, then checks `_ranked_candidates`.
     - No candidates >= topic_candidate_threshold -> schedule job (unchanged).
     - Candidates found -> cache the canonicalization and return them with a
       search_token; no job is created.
  2. Frontend either navigates to a candidate or re-POSTs with force=true and
     the search_token. The force path reuses the cached canonical title and
     embedding so the second call costs no LLM/embedding calls and is
     deterministic (LLM canonicalization is otherwise non-deterministic).

Redis is a cache-of-convenience, never a source of truth: a missing/expired
entry degrades to recomputing canonicalization, and every write/read failure is
logged and swallowed so a Redis outage never 500s course creation.
"""

import hashlib
import json
import logging
import math

import redis
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.course import Chapter, Course, CourseJob, Module

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.redis_url)
    return _client


def cache_key(topic_raw: str, user_id: int) -> str:
    digest = hashlib.sha256(f"{topic_raw}:{user_id}".encode("utf-8")).hexdigest()
    return f"knovolve:course:preview:{digest}"


def store_preview(key: str, canonical: str, embedding: list[float], topic_raw: str) -> None:
    """Persist a preview's canonicalization for a force re-POST.

    Failure is non-fatal: the token simply won't resolve and the force path
    recomputes canonicalization.
    """
    try:
        _get_client().set(
            key,
            json.dumps({"canonical": canonical, "embedding": embedding, "topic_raw": topic_raw}),
            ex=900,
        )
    except Exception:  # noqa: BLE001 - degrade, never fail course creation
        logger.warning("failed to cache course preview for %r; force will recompute",
                       topic_raw, exc_info=True)


def load_preview(key: str) -> dict | None:
    """Return the cached {canonical, embedding, topic_raw} or None on
    miss/expiry/failure."""
    try:
        raw = _get_client().get(key)
    except Exception:  # noqa: BLE001
        logger.warning("failed to read course preview cache key %s", key, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        # `raw` is typed as the redis client's generic ResponseT; loading only
        # the concrete str/bytes values the client can actually return.
        payload = json.loads(raw) if isinstance(raw, (str, bytes)) else None
    except (ValueError, TypeError):
        logger.warning("corrupt course preview cache payload for %s", key)
        return None
    if not isinstance(payload, dict) or "canonical" not in payload or "embedding" not in payload:
        logger.warning("unexpected course preview cache shape for %s", key)
        return None
    return payload


def clear_preview(key: str) -> None:
    try:
        _get_client().delete(key)
    except Exception:  # noqa: BLE001
        logger.warning("failed to delete course preview cache key %s", key, exc_info=True)


def _similarity(row) -> tuple[Course | CourseJob, float]:
    model, distance = row
    d = float(distance)
    # A zero/null embedding yields a NaN cosine distance; every comparison
    # with NaN is False, so an unguarded `similarity >= threshold` would
    # wrongly admit degenerate rows as candidates. Treat non-finite distances
    # as a guaranteed miss instead.
    if not math.isfinite(d):
        return model, float("-inf")
    return model, 1.0 - d


def _published_candidates(db: Session, embedding: list[float],
                          limit: int, threshold: float) -> list[dict]:
    rows = db.execute(
        select(Course, Course.topic_embedding.cosine_distance(embedding))
        .order_by(Course.topic_embedding.cosine_distance(embedding))
        .limit(64)
    ).all()
    out: list[dict] = []
    for row in rows:
        course, similarity = _similarity(row)
        if similarity < threshold:
            continue
        if len(out) >= limit:
            break
        module_count = db.scalar(
            select(func.count(Module.id)).where(Module.course_id == course.id)
        ) or 0
        chapter_count = db.scalar(
            select(func.count(Chapter.id)).where(
                Chapter.module_id.in_(select(Module.id).where(Module.course_id == course.id))
            )
        ) or 0
        out.append({
            "id": course.id,
            "topic_slug": course.topic_slug,
            "topic_raw": course.topic_raw,
            "similarity": round(similarity, 4),
            "status": course.status,
            "course_url": f"/courses/{course.topic_slug}",
            "module_count": module_count,
            "chapter_count": chapter_count,
        })
    return out


def _active_job_candidates(db: Session, embedding: list[float],
                           limit: int, threshold: float) -> list[dict]:
    rows = db.execute(
        select(CourseJob, CourseJob.topic_embedding.cosine_distance(embedding))
        .where(CourseJob.status.in_(["pending", "running"]))
        .order_by(CourseJob.topic_embedding.cosine_distance(embedding))
        .limit(64)
    ).all()
    out: list[dict] = []
    for row in rows:
        job, similarity = _similarity(row)
        if similarity < threshold:
            continue
        if len(out) >= limit:
            break
        out.append({
            "id": job.id,
            "topic_slug": job.topic_slug,
            "topic_raw": job.topic_raw,
            "similarity": round(similarity, 4),
            "status": job.status,
            "course_url": None,
            "module_count": None,
            "chapter_count": None,
        })
    return out


def ranked_candidates(db: Session, embedding: list[float],
                      limit: int, threshold: float) -> list[dict]:
    """Up to `limit` near courses, published courses before in-flight jobs,
    each with similarity >= threshold, descending similarity."""
    courses = _published_candidates(db, embedding, limit, threshold)
    if len(courses) >= limit:
        return courses
    jobs = _active_job_candidates(db, embedding, limit - len(courses), threshold)
    return courses + jobs