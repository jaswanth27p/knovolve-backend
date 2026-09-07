from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def _auth_headers():
    client.post("/auth/register", json={"email": "course-user@example.com", "password": "pw123456"})
    tokens = client.post("/auth/login", json={"email": "course-user@example.com", "password": "pw123456"}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}

def test_post_courses_requires_auth():
    resp = client.post("/courses", json={"topic": "Rust"})
    assert resp.status_code == 401

def test_post_courses_enqueues_job():
    headers = _auth_headers()
    with patch("app.routes.courses._canonicalize", return_value="Elixir"), \
         patch("app.routes.courses.find_existing", return_value=None), \
         patch("app.routes.courses.embed", return_value=[0.0] * 1024), \
         patch("app.routes.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "Elixir"}, headers=headers)
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"
    mock_delay.assert_called_once()

def test_post_courses_concurrent_same_topic_attaches_to_existing_job():
    headers = _auth_headers()
    from app.models.course import CourseJob
    fake_job = CourseJob(id=123, topic_slug="haskell", topic_raw="Haskell", status="pending")
    with patch("app.routes.courses._canonicalize", return_value="Haskell"), \
         patch("app.routes.courses.embed", return_value=[0.0] * 1024), \
         patch("app.routes.courses.find_existing", return_value=fake_job):
        resp = client.post("/courses", json={"topic": "haskell"}, headers=headers)
    assert resp.status_code == 202
    assert resp.json()["job_id"] == 123

def test_post_courses_dedups_on_canonical_embedding():
    """fix #1: dedup must compare embeddings of the canonical title, not the raw
    user string. Route canonicalizes first, embeds canonical, and the persisted
    job carries the canonical slug + canonical embedding."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with patch("app.routes.courses._canonicalize", return_value="Elixir") as mock_can, \
         patch("app.routes.courses.embed", return_value=[0.5] * 1024) as mock_embed, \
         patch("app.routes.courses.find_existing", return_value=None) as mock_find, \
         patch("app.routes.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "i want to learn elixir"}, headers=headers)
    assert resp.status_code == 202
    mock_can.assert_called_once_with("i want to learn elixir")
    mock_embed.assert_called_once_with("Elixir")
    assert mock_find.call_args.args[0] == [0.5] * 1024
    mock_delay.assert_called_once()

    with SessionLocal() as db:
        job = db.query(CourseJob).filter_by(topic_slug="elixir").one()
        assert job.topic_embedding == [0.5] * 1024

def test_post_courses_canonicalize_failure_falls_back_to_raw():
    """An LLM flap on canonicalization must not 500 the request: falling back to
    the raw topic keeps the flow functional (just with degraded dedup)."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with patch("app.routes.courses._canonicalize", side_effect=RuntimeError("llm down")), \
         patch("app.routes.courses.embed", return_value=[0.6] * 1024), \
         patch("app.routes.courses.find_existing", return_value=None), \
         patch("app.routes.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "Elixir"}, headers=headers)
    assert resp.status_code == 202
    mock_delay.assert_called_once()
    with SessionLocal() as db:
        job = db.query(CourseJob).filter_by(topic_slug="elixir").one()
        assert job.topic_embedding == [0.6] * 1024

def test_post_courses_recovers_when_concurrent_job_claims_slug():
    """TOCTOU backstop: find_existing misses an active job (true race), the
    insert hits the partial unique index on active topic_slug, and the route
    must recover by attaching to the job that won the race."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with SessionLocal() as db:
        job = CourseJob(topic_slug="elixir", topic_raw="Elixir", topic_embedding=[0.0] * 1024,
                        status="running", created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    with patch("app.routes.courses._canonicalize", return_value="Elixir"), \
         patch("app.routes.courses.embed", return_value=[0.7] * 1024), \
         patch("app.routes.courses.find_existing", return_value=None), \
         patch("app.routes.courses.run_course_creation_job.delay") as mock_delay:
        resp = client.post("/courses", json={"topic": "elixir study guide"}, headers=headers)
    assert resp.status_code == 202
    assert resp.json()["job_id"] == job_id
    mock_delay.assert_not_called()
    with SessionLocal() as db:
        assert db.query(CourseJob).filter_by(topic_slug="elixir", status="running").count() == 1

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
                         topic_embedding=[0.0] * 1024, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)

        job = CourseJob(topic_slug="job-status-test", topic_raw="x", topic_embedding=[0.0] * 1024,
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


def test_post_courses_existing_course_returns_200_not_202():
    """A completed read (course already exists) must be a 200, not the 202 the
    enqueue path uses."""
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import Course
    with SessionLocal() as db:
        course = Course(topic_slug="typescript", topic_raw="TypeScript",
                        topic_embedding=[0.9] + [0.0] * 1023,
                        created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        course_id = course.id

    with patch("app.routes.courses._canonicalize", return_value="TypeScript"), \
         patch("app.routes.courses.embed", return_value=[0.9] + [0.0] * 1023), \
         patch("app.routes.courses.find_existing", return_value=course):
        resp = client.post("/courses", json={"topic": "typescript"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "exists"
    assert body["course"]["id"] == course_id


def test_retry_failed_job_reenqueues():
    headers = _auth_headers()
    from datetime import datetime, timezone
    from app.db import SessionLocal
    from app.models.course import CourseJob
    with SessionLocal() as db:
        job = CourseJob(topic_slug="retry-topic", topic_raw="Retry Topic",
                        topic_embedding=[1.0] + [0.0] * 1023, status="failed",
                        error="old raw error text",
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    with patch("app.routes.courses.find_existing", return_value=None), \
         patch("app.routes.courses.run_course_creation_job.delay") as mock_delay:
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
                        topic_embedding=[1.0] + [0.0] * 1023, status="running",
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
                        topic_embedding=[1.0] + [0.0] * 1023,
                        created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        course_id = course.id
        job = CourseJob(topic_slug="elixir", topic_raw="Elixir",
                        topic_embedding=[1.0] + [0.0] * 1023, status="failed",
                        error="generic msg",
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    with patch("app.routes.courses.find_existing", return_value=course), \
         patch("app.routes.courses.run_course_creation_job.delay") as mock_delay:
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
                    topic_embedding=[0.0] * 1024, created_at=now)
        c2 = Course(topic_slug="pub-2", topic_raw="pub 2",
                    topic_embedding=[0.0] * 1024, created_at=now)
        db.add_all([c1, c2])
        db.commit()
        db.add(Module(course_id=c1.id, title="M", objective="o", order=1))
        db.add(Module(course_id=c2.id, title="M", objective="o", order=1))
        db.commit()

    # User opens pub-2 -> it becomes tracked -> must be excluded.
    client.get("/courses/pub-2", headers=headers)

    resp = client.get("/courses", headers=headers)
    body = resp.json()
    slugs = {c["topic_slug"] for c in body}
    assert "pub-1" in slugs
    assert "pub-2" not in slugs
    row = next(c for c in body if c["topic_slug"] == "pub-1")
    assert row["module_count"] == 1
