from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models.course import Course

client = TestClient(app)


def _auth_headers():
    client.post("/auth/register", json={"email": "ext-user@example.com", "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": "ext-user@example.com", "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}


def _make_course():
    with SessionLocal() as db:
        course = Course(topic_slug="ext-api-course", topic_raw="Ext API",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course


def test_extend_endpoints_require_auth():
    _make_course()
    assert client.post("/courses/ext-api-course/extensions", json={"message": "x"}).status_code == 401


def test_create_poll_list_delete_flow():
    headers = _auth_headers()
    _make_course()
    with patch("app.routes.course_extensions.run_course_extension_job.delay") as mock_delay:
        resp = client.post("/courses/ext-api-course/extensions",
                           json={"message": "add tcp"}, headers=headers)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == "pending"
    mock_delay.assert_called_once_with(job_id)

    resp = client.get(f"/courses/ext-api-course/extensions/jobs/{job_id}",
                      headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"

    resp = client.get("/courses/ext-api-course/extensions/chapters", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == []

    from app.db import SessionLocal as S
    from app.models.course_extension import CourseExtensionJob as J
    from app.models.user import User
    from app.services import course_extension as svc
    from sqlalchemy import select
    with S() as db:
        course = db.query(Course).filter_by(topic_slug="ext-api-course").one()
        uid = db.scalar(select(User.id).limit(1))
        assert uid is not None
        added = svc.append_chapters(db, uid, course, [{"title": "TCP", "objective": "o"}])
        job = db.scalar(select(J).where(J.id == job_id))
        assert job is not None
        job.status = "succeeded"
        job.result = added
        job.updated_at = datetime.now(timezone.utc)
        db.commit()

    resp = client.get("/courses/ext-api-course/extensions/chapters", headers=headers)
    assert resp.json()[0]["title"] == "TCP"

    resp = client.delete(f"/courses/ext-api-course/extensions/chapters/{resp.json()[0]['id']}",
                         headers=headers)
    assert resp.status_code == 204
    resp = client.get("/courses/ext-api-course/extensions/chapters", headers=headers)
    assert resp.json() == []


def test_concurrent_run_rejected():
    headers = _auth_headers()
    _make_course()
    with patch("app.routes.course_extensions.run_course_extension_job.delay"):
        first = client.post("/courses/ext-api-course/extensions",
                            json={"message": "x"}, headers=headers)
    assert first.status_code == 202
    with patch("app.routes.course_extensions.run_course_extension_job.delay"):
        second = client.post("/courses/ext-api-course/extensions",
                             json={"message": "y"}, headers=headers)
    assert second.status_code == 409


def test_other_user_cannot_see_or_delete():
    headers = _auth_headers()
    _make_course()
    client.post("/auth/register", json={"email": "ext-other@example.com", "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": "ext-other@example.com", "password": "pw123456"})
    other_headers = {"Authorization": f"Bearer {resp.cookies['access_token']}"}
    from app.db import SessionLocal as S
    from app.models.user import User
    from app.services import course_extension as svc
    from sqlalchemy import select
    with S() as db:
        course = db.query(Course).filter_by(topic_slug="ext-api-course").one()
        uid = db.scalar(select(User.id).limit(1))
        assert uid is not None
        added = svc.append_chapters(db, uid, course, [{"title": "Private", "objective": "o"}])
        db.commit()
        cid = added[0]["chapter_id"]
    resp = client.get("/courses/ext-api-course/extensions/chapters", headers=headers)
    assert [c["title"] for c in resp.json()] == ["Private"]
    resp = client.get("/courses/ext-api-course/extensions/chapters", headers=other_headers)
    assert resp.json() == []
    resp = client.delete(f"/courses/ext-api-course/extensions/chapters/{cid}", headers=other_headers)
    assert resp.status_code == 404