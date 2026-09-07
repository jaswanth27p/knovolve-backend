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
    with patch("app.routes.courses.find_existing", return_value=None), \
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
    with patch("app.routes.courses.find_existing", return_value=fake_job):
        resp = client.post("/courses", json={"topic": "haskell"}, headers=headers)
    assert resp.status_code == 202
    assert resp.json()["job_id"] == 123

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
