import uuid
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.user import User

client = TestClient(app)


def _auth_for(email: str) -> dict:
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": email, "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}


def _get_user_id(email: str) -> int:
    with SessionLocal() as db:
        return db.query(User).filter_by(email=email).one().id


def _attempt_row(user_id: int, score: float, created_at: datetime, status: str = "graded") -> None:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        slug = f"ac-{user_id}-{uuid.uuid4().hex}"
        course = Course(topic_slug=slug, topic_raw=slug,
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course); db.flush()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module); db.flush()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
        db.add(chapter); db.flush()
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                 outline=[], created_at=now, updated_at=now)
        db.add(content); db.flush()
        assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                                status="ready", created_at=now, updated_at=now)
        db.add(assignment); db.flush()
        db.add(AssignmentAttempt(assignment_id=assignment.id, user_id=user_id, status=status,
                                 overall_score=score, created_at=created_at, updated_at=created_at))
        db.commit()


def test_activity_requires_auth():
    resp = client.get("/me/activity")
    assert resp.status_code == 401


def test_activity_returns_only_own_graded_attempts_in_window():
    headers = _auth_for("act-a@example.com")
    uid = _get_user_id("act-a@example.com")
    now = datetime.now(timezone.utc)
    _attempt_row(uid, 0.9, now - timedelta(days=1))
    _attempt_row(uid, 0.4, now - timedelta(days=2))

    resp = client.get("/me/activity?days=14", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["days"] == 14
    assert len(body["events"]) == 2
    passed = [e for e in body["events"] if e["passed"]]
    assert len(passed) == 1 and passed[0]["score"] == 0.9

    ats = [e["at"] for e in body["events"]]
    assert ats == sorted(ats)


def test_activity_excludes_other_users_and_pending_failed_attempts():
    headers_a = _auth_for("act-b@example.com")
    headers_b = _auth_for("act-c@example.com")
    uid_a = _get_user_id("act-b@example.com")
    now = datetime.now(timezone.utc)
    _attempt_row(uid_a, 0.8, now - timedelta(days=1))                       # own graded -> included
    _attempt_row(uid_a, 0.8, now - timedelta(days=1), status="failed")      # own but failed -> excluded

    resp_a = client.get("/me/activity?days=14", headers=headers_a)
    assert len(resp_a.json()["events"]) == 1

    resp_b = client.get("/me/activity?days=14", headers=headers_b)
    assert len(resp_b.json()["events"]) == 0


def test_activity_window_is_days_plus_one():
    headers = _auth_for("act-d@example.com")
    uid = _get_user_id("act-d@example.com")
    now = datetime.now(timezone.utc)
    _attempt_row(uid, 0.7, now - timedelta(days=14))    # 14 UTC days back -> included (margin)
    _attempt_row(uid, 0.7, now - timedelta(days=15))    # 15 UTC days back -> excluded

    resp = client.get("/me/activity?days=14", headers=headers)
    body = resp.json()
    assert len(body["events"]) == 1


def test_activity_empty():
    headers = _auth_for("act-e@example.com")
    resp = client.get("/me/activity?days=14", headers=headers)
    assert resp.json()["events"] == []
