from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def _auth_headers():
    client.post("/auth/register", json={"email": "course-user@example.com", "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": "course-user@example.com", "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}

def test_post_courses_requires_auth():
    resp = client.post("/courses", json={"topic": "Rust"})
    assert resp.status_code == 401

def test_post_courses_enqueues_job():
    headers = _auth_headers()
    with patch("app.services.courses._canonicalize", return_value="Elixir"), \
         patch("app.services.courses.embed", return_value=[0.0] * 2048), \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "Elixir"}, headers=headers)
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"
    mock_delay.assert_called_once()

def test_post_courses_preview_surfaces_inflight_job_as_candidate():
    """A pending/running job matching the topic must appear as a candidate
    rather than auto-attaching — the user decides."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with SessionLocal() as db:
        job = CourseJob(topic_slug="haskell", topic_raw="Haskell",
                        topic_embedding=[1.0] + [0.0] * 2047, status="running",
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    with patch("app.services.courses._canonicalize", return_value="Haskell"), \
         patch("app.services.courses.embed", return_value=[0.95] + [0.0] * 2047), \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "haskell"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "similar"
    assert body["search_token"]
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["id"] == job_id
    assert body["candidates"][0]["status"] == "running"
    assert body["candidates"][0]["course_url"] is None
    mock_delay.assert_not_called()

def test_post_courses_dedups_on_canonical_embedding():
    """Canonicalization + embedding happen once up front, are cached for a
    force re-POST, and candidate ranking runs on the canonical embedding."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with patch("app.services.courses._canonicalize", return_value="Elixir") as mock_can, \
         patch("app.services.courses.embed", return_value=[0.5] * 2048) as mock_embed, \
         patch("app.services.courses.course_preview.store_preview") as mock_store, \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "i want to learn elixir"}, headers=headers)
    assert resp.status_code == 202
    mock_can.assert_called_once_with("i want to learn elixir")
    mock_embed.assert_called_once_with("Elixir")
    assert mock_store.call_args.args[2] == [0.5] * 2048  # canonical embedding cached
    mock_delay.assert_called_once()

    with SessionLocal() as db:
        job = db.query(CourseJob).filter_by(topic_slug="elixir").one()
        assert job.topic_embedding == [0.5] * 2048
        assert job.allow_duplicate is False

def test_post_courses_canonicalize_failure_falls_back_to_raw():
    """An LLM flap on canonicalization must not 500 the request: falling back to
    the raw topic keeps the flow functional (just with degraded dedup)."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with patch("app.services.courses._canonicalize", side_effect=RuntimeError("llm down")), \
         patch("app.services.courses.embed", return_value=[0.5] * 2048), \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "Elixir"}, headers=headers)
    assert resp.status_code == 202
    mock_delay.assert_called_once()
    with SessionLocal() as db:
        job = db.query(CourseJob).filter_by(topic_slug="elixir").one()
        assert job.topic_embedding == [0.5] * 2048

def test_post_courses_recovers_when_concurrent_job_claims_slug():
    """TOCTOU backstop: the candidate step misses an active job (true race, or
    embedding below the candidate threshold), the insert hits the partial unique
    index on active topic_slug, and the route must recover by attaching to the
    job that won the race."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with SessionLocal() as db:
        job = CourseJob(topic_slug="elixir", topic_raw="Elixir", topic_embedding=[0.0] * 2048,
                        status="running", created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    with patch("app.services.courses._canonicalize", return_value="Elixir"), \
         patch("app.services.courses.embed", return_value=[0.5] * 2048), \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "elixir study guide"}, headers=headers)
    assert resp.status_code == 202
    assert resp.json()["job_id"] == job_id
    mock_delay.assert_not_called()
    with SessionLocal() as db:
        assert db.query(CourseJob).filter_by(topic_slug="elixir", status="running").count() == 1

def test_post_courses_blocks_second_job_while_one_is_ongoing():
    """A user can only have one course generation in flight at a time —
    the /learn UI blocks the create form while a job is running, so a
    second request for this user must attach to the existing job rather
    than starting a new one, regardless of topic."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    from app.models.user import User
    with SessionLocal() as db:
        user_id = db.query(User).filter_by(email="course-user@example.com").one().id
        job = CourseJob(topic_slug="elixir", topic_raw="Elixir", topic_embedding=[0.0] * 2048,
                        status="running", created_by_user_id=user_id,
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        ongoing_job_id = job.id

    with patch("app.services.courses._canonicalize") as mock_can, \
         patch("app.services.courses.embed") as mock_embed, \
         patch("app.services.courses.find_existing") as mock_find, \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "Rust"}, headers=headers)
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"
    assert resp.json()["job_id"] == ongoing_job_id
    mock_can.assert_not_called()
    mock_embed.assert_not_called()
    mock_find.assert_not_called()
    mock_delay.assert_not_called()

def test_get_job_status():
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import Course, CourseJob
    with SessionLocal() as db:
        # course_id has an FK to courses.id, so a real Course row must exist
        # before a job can reference it (the brief's literal course_id=1
        # assumed an empty-table row 1 without inserting one first).
        course = Course(topic_slug="job-status-test-course", topic_raw="x",
                         topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)

        job = CourseJob(topic_slug="job-status-test", topic_raw="x", topic_embedding=[0.0] * 2048,
                          status="succeeded", course_id=course.id, created_at=datetime.now(timezone.utc),
                          updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    resp = client.get(f"/courses/jobs/{job_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "succeeded"

def test_get_course_by_slug_not_found():
    headers = _auth_headers()
    resp = client.get("/courses/does-not-exist", headers=headers)
    assert resp.status_code == 404


def test_post_courses_existing_course_returns_similar_list():
    """A finished course close to the topic is offered as a candidate (200)
    instead of silently auto-navigating — nothing is merged without consent."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import Course
    with SessionLocal() as db:
        course = Course(topic_slug="typescript", topic_raw="TypeScript",
                        topic_embedding=[0.9] + [0.0] * 2047,
                        created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        course_id = course.id

    with patch("app.services.courses._canonicalize", return_value="TypeScript"), \
         patch("app.services.courses.embed", return_value=[0.9] + [0.0] * 2047), \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "typescript"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "similar"
    assert len(body["candidates"]) == 1
    cand = body["candidates"][0]
    assert cand["id"] == course_id
    assert cand["course_url"] == "/courses/typescript"
    assert isinstance(cand["module_count"], int)
    mock_delay.assert_not_called()

def test_post_courses_force_generates_new_course():
    """Force reuses the cached canonicalization and schedules a job with
    allow_duplicate=True, bypassing the similarity preview."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import Course, CourseJob
    with SessionLocal() as db:
        course = Course(topic_slug="backend-development", topic_raw="Backend Development",
                        topic_embedding=[0.9] + [0.0] * 2047,
                        created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()

    with patch("app.services.courses.course_preview.load_preview",
               return_value={"canonical": "Python Backend Development",
                             "embedding": [0.8] + [0.0] * 2047,
                             "topic_raw": "python backend"}), \
         patch("app.services.courses.course_preview.clear_preview") as mock_clear, \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "python backend", "force": True,
                                             "search_token": "knovolve:course:preview:abc"},
                           headers=headers)
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"
    mock_delay.assert_called_once()
    mock_clear.assert_called_once_with("knovolve:course:preview:abc")

    with SessionLocal() as db:
        job = db.query(CourseJob).filter_by(topic_slug="python-backend-development").one()
        assert job.allow_duplicate is True
        assert job.topic_raw == "python backend"

def test_post_courses_force_attaches_to_exact_duplicate():
    """Force bypasses similarity but not exact identity: a published course
    with the exact canonical slug is attached to (exists), not copied."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import Course
    with SessionLocal() as db:
        course = Course(topic_slug="typescript", topic_raw="TypeScript",
                        topic_embedding=[0.9] + [0.0] * 2047,
                        created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        course_id = course.id

    with patch("app.services.courses.course_preview.load_preview",
               return_value={"canonical": "TypeScript",
                             "embedding": [0.9] + [0.0] * 2047,
                             "topic_raw": "typescript"}), \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "typescript", "force": True,
                                             "search_token": "knovolve:course:preview:ty"},
                           headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "exists"
    assert resp.json()["course"]["id"] == course_id
    mock_delay.assert_not_called()

def test_post_courses_force_with_missing_token_recomputes():
    """An expired/missing search_token degrades to recomputing canonicalization
    rather than failing the request."""
    headers = _auth_headers()
    with patch("app.services.courses.course_preview.load_preview", return_value=None), \
         patch("app.services.courses._canonicalize", return_value="Elixir") as mock_can, \
         patch("app.services.courses.embed", return_value=[0.0] * 2048) as mock_embed, \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "Elixir", "force": True,
                                             "search_token": "knovolve:course:preview:expired"},
                           headers=headers)
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"
    mock_can.assert_called_once()
    mock_embed.assert_called_once()
    mock_delay.assert_called_once()


def test_retry_failed_job_reenqueues():
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with SessionLocal() as db:
        job = CourseJob(topic_slug="retry-topic", topic_raw="Retry Topic",
                        topic_embedding=[1.0] + [0.0] * 2047, status="failed",
                        error="old raw error text",
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    with patch("app.services.courses.find_existing", return_value=None), \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post(f"/courses/jobs/{job_id}/retry", headers=headers)
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"
    assert resp.json()["job_id"] == job_id
    mock_delay.assert_called_once()

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "pending"
        assert job.error is None


def test_retry_non_failed_job_returns_409():
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with SessionLocal() as db:
        job = CourseJob(topic_slug="running-topic", topic_raw="Running Topic",
                        topic_embedding=[1.0] + [0.0] * 2047, status="running",
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    resp = client.post(f"/courses/jobs/{job_id}/retry", headers=headers)
    assert resp.status_code == 409


def test_retry_attaches_course_created_elsewhere():
    """If a course for the failed topic now exists, retry must attach it instead
    of regenerating — no wasted generation."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import Course, CourseJob
    with SessionLocal() as db:
        course = Course(topic_slug="elixir", topic_raw="Elixir",
                        topic_embedding=[1.0] + [0.0] * 2047,
                        created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        course_id = course.id
        job = CourseJob(topic_slug="elixir", topic_raw="Elixir",
                        topic_embedding=[1.0] + [0.0] * 2047, status="failed",
                        error="generic msg",
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    with patch("app.services.courses.find_existing", return_value=course), \
         patch("app.services.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post(f"/courses/jobs/{job_id}/retry", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "exists"
    mock_delay.assert_not_called()

    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.course_id == course_id
        assert job.error is None


def test_get_public_courses_requires_auth():
    resp = client.get("/courses")
    assert resp.status_code == 401


def test_get_public_courses_excludes_tracked():
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import Course, Module
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        c1 = Course(topic_slug="pub-1", topic_raw="pub 1",
                    topic_embedding=[0.0] * 2048, created_at=now)
        c2 = Course(topic_slug="pub-2", topic_raw="pub 2",
                    topic_embedding=[0.0] * 2048, created_at=now)
        db.add_all([c1, c2])
        db.commit()
        db.add(Module(course_id=c1.id, title="M", objective="o", order=1))
        db.add(Module(course_id=c2.id, title="M", objective="o", order=1))
        db.commit()

    # User opens pub-2 -> it becomes tracked -> must be excluded.
    client.get("/courses/pub-2", headers=headers)

    resp = client.get("/courses", headers=headers)
    body = resp.json()
    slugs = {c["topic_slug"] for c in body["items"]}
    assert "pub-1" in slugs
    assert "pub-2" not in slugs
    row = next(c for c in body["items"] if c["topic_slug"] == "pub-1")
    assert row["module_count"] == 1


def _make_public_course(slug: str, title: str):
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import Course
    with SessionLocal() as db:
        course = Course(topic_slug=slug, topic_raw=title,
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course.id


def test_get_public_courses_paginated_envelope():
    headers = _auth_headers()
    _make_public_course("pg-1", "Pg One")
    _make_public_course("pg-2", "Pg Two")
    _make_public_course("pg-3", "Pg Three")

    resp = client.get("/courses?page=1&limit=2", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["page"] == 1
    assert body["limit"] == 2
    assert body["total_pages"] == 2

    resp2 = client.get("/courses?page=2&limit=2", headers=headers)
    body2 = resp2.json()
    assert len(body2["items"]) == 1
    assert body2["total"] == 3


def test_get_public_courses_search_filters_by_title():
    headers = _auth_headers()
    _make_public_course("srch-rust", "Rust Basics")
    _make_public_course("srch-elixir", "Elixir Guide")

    resp = client.get("/courses?search=rust", headers=headers)
    body = resp.json()
    slugs = {c["topic_slug"] for c in body["items"]}
    assert slugs == {"srch-rust"}


def test_get_public_courses_sort_by_name_asc():
    headers = _auth_headers()
    _make_public_course("sort-zebra", "Zebra")
    _make_public_course("sort-alpha", "Alpha")

    resp = client.get("/courses?sort=name&order=asc", headers=headers)
    titles = [c["topic_raw"] for c in resp.json()["items"]]
    assert titles.index("Alpha") < titles.index("Zebra")
