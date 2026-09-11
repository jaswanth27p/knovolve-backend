from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.db import SessionLocal
from app.main import app
from app.models.course import Course, Module, Chapter
from app.models.course import Course as CourseModel
from app.models.export import CourseGenerationRun
from app.models.user import User
from app.services import generation as generation_service
from app.tasks.course_generation_task import run_course_generation_task

client = TestClient(app)


def _auth_headers(email):
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": email, "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}


def _make_course(slug):
    with SessionLocal() as db:
        course = Course(topic_slug=slug, topic_raw="Generation",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course.id


def _make_module_with_chapters(course_id, count=2):
    with SessionLocal() as db:
        module = Module(course_id=course_id, title="M", objective="o", order=1, scope="global")
        db.add(module)
        db.commit()
        for order in range(1, count + 1):
            db.add(Chapter(module_id=module.id, title=f"C{order}", objective="o", order=order, scope="global"))
        db.commit()


def test_queue_generation_returns_worker_action():
    headers = _auth_headers("generation-api@example.com")
    _make_course("generation-api-course")
    fake_run = SimpleNamespace(id=301, status="pending", total_units=2, completed_units=0,
                               unit_states=[], error=None)
    with patch("app.services.generation.queue_generation_run", return_value=(fake_run, "queued")), \
        patch("app.routes.generation.run_course_generation_task.delay") as mock_delay:
        resp = client.post("/courses/generation-api-course/generate", headers=headers)
    assert resp.status_code == 202
    assert resp.json()["action"] == "queued"
    mock_delay.assert_called_once_with(301)


def test_already_complete_generation_does_not_dispatch_worker():
    headers = _auth_headers("generation-complete@example.com")
    _make_course("generation-complete-course")
    fake_run = SimpleNamespace(id=302, status="succeeded", total_units=0, completed_units=0,
                               unit_states=[], error=None)
    with patch("app.services.generation.queue_generation_run", return_value=(fake_run, "already_complete")), \
        patch("app.routes.generation.run_course_generation_task.delay") as mock_delay:
        resp = client.post("/courses/generation-complete-course/generate", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["action"] == "already_complete"
    mock_delay.assert_not_called()


def test_generation_endpoints_require_auth():
    _make_course("generation-auth-course")
    assert client.post("/courses/generation-auth-course/generate").status_code == 401
    assert client.get("/courses/generation-auth-course/generation").status_code == 401
    assert client.get("/courses/generation-auth-course/readiness").status_code == 401


def test_readiness_returns_service_status():
    headers = _auth_headers("generation-readiness@example.com")
    _make_course("generation-readiness-course")
    readiness = {
        "status": "complete",
        "global_content_ready": True,
        "additional_content_ready": True,
        "versions_ready": True,
        "assignments_ready": True,
    }
    with patch("app.services.generation.readiness_summary", return_value=readiness):
        resp = client.get("/courses/generation-readiness-course/readiness", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == readiness


def _seed_pending_run(slug="generation-task-course"):
    with SessionLocal() as db:
        db.add(User(id=91, email="generation-task@example.com", password_hash="x"))
        db.commit()
        course_id = _make_course(slug)
        _make_module_with_chapters(course_id)
        course = db.get(CourseModel, course_id)
        assert course is not None
        run, action = generation_service.queue_generation_run(db, 91, course)
        assert action == "queued"
        return run.id


def test_worker_executes_planned_units_and_completes():
    run_id = _seed_pending_run()
    with patch("app.tasks.course_generation_task.ensure_chapter_content") as mock_content:
        run_course_generation_task(run_id)  # pyright: ignore[reportCallIssue]

    assert mock_content.call_count == 2
    with SessionLocal() as db:
        run = db.get(CourseGenerationRun, run_id)
        assert run is not None
        assert run.status == "succeeded"
        assert run.completed_units == 2
        assert [u["status"] for u in run.unit_states] == ["done", "done"]


def test_unit_failure_marks_run_failed_without_losing_progress():
    run_id = _seed_pending_run("generation-failure-course")
    with patch("app.tasks.course_generation_task.ensure_chapter_content",
              side_effect=[ValueError("LLM failed"), None]):
        run_course_generation_task(run_id)  # pyright: ignore[reportCallIssue]

    with SessionLocal() as db:
        run = db.get(CourseGenerationRun, run_id)
        assert run is not None
        assert run.status == "failed"
        assert run.completed_units == 2
        assert [u["status"] for u in run.unit_states] == ["failed", "done"]
        assert run.error == "Course generation failed. Please try again."
