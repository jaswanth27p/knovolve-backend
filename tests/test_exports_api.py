from datetime import datetime, timezone
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.db import SessionLocal
from app.main import app
from app.models.course import Course
from app.models.export import ExportJob
from app.models.user import User

client = TestClient(app)


def _auth_headers(email):
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": email, "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}


def _make_course(slug):
    with SessionLocal() as db:
        course = Course(topic_slug=slug, topic_raw="API",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course.id


def test_export_endpoints_require_auth():
    _make_course("export-auth-course")
    assert client.post("/courses/export-auth-course/exports", json={"kind": "course"}).status_code == 401
    assert client.get("/courses/export-auth-course/exports").status_code == 401


def test_create_export_queues_worker_and_lists_history():
    headers = _auth_headers("export-api@example.com")
    _make_course("export-api-course")
    with patch("app.services.exports.require_export_ready"), \
        patch("app.tasks.export_tasks.run_export_task.delay") as mock_delay, \
        patch("app.services.exports.list_export_jobs", return_value=[]):
        created = client.post("/courses/export-api-course/exports", json={"kind": "course"}, headers=headers)
        assert created.status_code == 202
        mock_delay.assert_called_once_with(created.json()["id"])
        listed = client.get("/courses/export-api-course/exports", headers=headers)
        assert listed.status_code == 200
        assert listed.json() == []


def test_export_detail_is_user_and_course_scoped():
    headers = _auth_headers("export-scoped@example.com")
    course_id = _make_course("export-scoped-course")
    with SessionLocal() as db:
        db.add(User(id=999999, email="other-export@example.com", password_hash="x"))
        db.commit()
        other_job = ExportJob(course_id=course_id, user_id=999999, kind="course", status="pending",
                              created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(other_job)
        db.commit()
        other_id = other_job.id
    resp = client.get(f"/courses/export-scoped-course/exports/{other_id}", headers=headers)
    assert resp.status_code == 404


def test_download_redirects_only_for_successful_exports():
    headers = _auth_headers("export-download@example.com")
    _make_course("export-download-course")
    with patch("app.services.exports.presigned_export_url", return_value="https://signed.example/a.pdf"):
        resp = client.get("/courses/export-download-course/exports/1/download", headers=headers, follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "https://signed.example/a.pdf"


def test_retry_export_requeues_failed_job_and_dispatches_worker():
    headers = _auth_headers("export-retry@example.com")
    course_id = _make_course("export-retry-course")
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        user = db.query(User).filter_by(email="export-retry@example.com").one()
        failed = ExportJob(course_id=course_id, user_id=user.id, kind="course", status="failed",
                           created_at=now, updated_at=now)
        db.add(failed)
        db.commit()
        db.refresh(failed)
        failed_id = failed.id
    with patch("app.services.exports.require_export_ready"), \
        patch("app.tasks.export_tasks.run_export_task.delay") as mock_delay:
        resp = client.post(f"/courses/export-retry-course/exports/{failed_id}/retry", headers=headers)
    assert resp.status_code == 202
    new_id = resp.json()["id"]
    assert new_id != failed_id
    mock_delay.assert_called_once_with(new_id)


def test_download_url_returns_presigned_json_for_browser_clients():
    headers = _auth_headers("export-download-url@example.com")
    _make_course("export-download-url-course")
    with patch("app.services.exports.presigned_export_url", return_value="https://signed.example/a.pdf"):
        resp = client.get("/courses/export-download-url-course/exports/1/download-url", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"url": "https://signed.example/a.pdf"}
