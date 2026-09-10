import math
from datetime import datetime, timezone

from app.db import SessionLocal
from app.models.course import Course, CourseJob, Module, Chapter
from app.services import course_preview


def _unit(a: float) -> list[float]:
    """A unit-norm embedding whose cosine with [1.0, 0, ...] is exactly `a`."""
    return [a, math.sqrt(1 - a ** 2)] + [0.0] * (2048 - 2)


class _FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value
        return True

    def delete(self, *keys):
        removed = 0
        for k in keys:
            if self.data.pop(k, None) is not None:
                removed += 1
        return removed


def _with_fake_redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(course_preview, "_get_client", lambda: fake)
    return fake


def test_cache_key_scopes_by_raw_and_user():
    k1 = course_preview.cache_key("python backend", 1)
    k2 = course_preview.cache_key("python backend", 2)
    k3 = course_preview.cache_key("python  backend", 1)
    assert k1 != k2
    assert k1 != k3
    assert k1.startswith("knovolve:course:preview:")


def test_cache_roundtrip(monkeypatch):
    fake = _with_fake_redis(monkeypatch)
    key = "knovolve:course:preview:rt"
    course_preview.store_preview(key, "Python Backend Development", [0.1] * 2048, "python backend")
    assert fake.data[key] is not None

    loaded = course_preview.load_preview(key)
    assert loaded == {"canonical": "Python Backend Development",
                      "embedding": [0.1] * 2048, "topic_raw": "python backend"}

    course_preview.clear_preview(key)
    assert course_preview.load_preview(key) is None


def test_load_preview_returns_none_for_missing_or_corrupt(monkeypatch):
    fake = _with_fake_redis(monkeypatch)
    assert course_preview.load_preview("knovolve:course:preview:missing") is None

    fake.data["knovolve:course:preview:bad"] = "not json"
    assert course_preview.load_preview("knovolve:course:preview:bad") is None

    fake.data["knovolve:course:preview:shape"] = '{"canonical":"x"}'
    assert course_preview.load_preview("knovolve:course:preview:shape") is None


def test_load_preview_survives_redis_failure(monkeypatch):
    def boom(_):
        raise RuntimeError("redis down")

    monkeypatch.setattr(course_preview, "_get_client", boom)
    assert course_preview.load_preview("knovolve:course:preview:down") is None

    def boom_set(*_args, **_kwargs):
        raise RuntimeError("redis down")

    monkeypatch.setattr(course_preview, "_get_client", boom_set)
    # store/clear swallow failures too — course creation must never 500 on redis.
    course_preview.store_preview("knovolve:course:preview:down", "x", [0.0] * 2048, "x")
    course_preview.clear_preview("knovolve:course:preview:down")


def test_ranked_candidates_orders_courses_first_and_caps():
    query = [1.0] + [0.0] * 2047
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        best = Course(topic_slug="best", topic_raw="Best", topic_embedding=_unit(0.95),
                      created_at=now)
        mid = Course(topic_slug="mid", topic_raw="Mid", topic_embedding=_unit(0.75),
                     created_at=now)
        below = Course(topic_slug="below", topic_raw="Below", topic_embedding=_unit(0.60),
                       created_at=now)
        db.add_all([best, mid, below])
        db.commit()
        db.refresh(best)
        db.refresh(mid)
        module = Module(course_id=best.id, title="m", objective="o", order=1)
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="c", objective="o", order=1)
        db.add(chapter)
        job = CourseJob(topic_slug="genning", topic_raw="Genning", topic_embedding=_unit(0.90),
                        status="running", created_at=now, updated_at=now)
        db.add(job)
        db.commit()

    with SessionLocal() as db:
        results = course_preview.ranked_candidates(db, query, limit=3, threshold=0.70)

    assert len(results) == 3
    slugs = [r["topic_slug"] for r in results]
    assert slugs[0] == "best"
    assert slugs[1] == "mid"              # published courses fill the cap first
    assert slugs[2] == "genning"
    assert "below" not in slugs           # 0.60 < 0.70

    c0 = results[0]
    assert c0["course_url"] == "/courses/best"
    assert c0["module_count"] == 1
    assert c0["chapter_count"] == 1
    assert results[2]["course_url"] is None
    assert results[2]["similarity"] == 0.9