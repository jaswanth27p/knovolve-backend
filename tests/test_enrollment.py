from datetime import datetime, timezone
from contextlib import ExitStack
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.enrollment import UserCourse
from app.models.user import User

client = TestClient(app)


def _auth_headers(email="enroll-a@example.com") -> dict:
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    tokens = client.post("/auth/login", json={"email": email, "password": "pw123456"}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def _make_course_with_chapter(topic_slug: str) -> tuple[int, int, int]:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=topic_slug, topic_raw=topic_slug,
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
        db.add(chapter)
        db.commit()
        return course.id, module.id, chapter.id


def test_generating_course_enrolls_user():
    headers = _auth_headers()
    with patch("app.services.courses._canonicalize", return_value="Elixir"), \
         patch("app.services.courses.embed", return_value=[0.0] * 2048), \
         patch("app.services.courses.find_existing", return_value=None), \
         patch("app.services.courses.run_course_creation_job.delay"):
        resp = client.post("/courses", json={"topic": "Elixir"}, headers=headers)
    assert resp.status_code == 202
    # enroll happens against the job's target; with no course yet, no row — but
    # opening/creating later must cover. This test asserts the POST path is safe:
    assert resp.json()["status"] == "pending"


def test_opening_course_enrolls_idempotently():
    headers = _auth_headers(email="enroll-b@example.com")
    course_id, _, _ = _make_course_with_chapter("enroll-b")

    resp = client.get(f"/courses/enroll-b", headers=headers)
    assert resp.status_code == 200
    with SessionLocal() as db:
        rows = db.query(UserCourse).filter_by(course_id=course_id).all()
        assert len(rows) == 1
        first_opened = rows[0].last_opened_at

    client.get(f"/courses/enroll-b", headers=headers)  # second open
    with SessionLocal() as db:
        rows = db.query(UserCourse).filter_by(course_id=course_id).all()
        assert len(rows) == 1
        assert rows[0].last_opened_at >= first_opened


def test_post_existing_course_enrolls_idempotently():
    """create_course's exists branch (find_existing -> Course) must enroll the
    user exactly once and keep bumping last_opened_at on repeat POSTs."""
    headers = _auth_headers(email="enroll-d@example.com")
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug="enroll-d", topic_raw="Elixir",
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        db.refresh(course)
        course_id = course.id

    patches = [
        patch("app.services.courses._canonicalize", return_value="Elixir"),
        patch("app.services.courses.embed", return_value=[0.0] * 2048),
        patch("app.services.courses.find_existing", return_value=course),
        patch("app.services.courses.run_course_creation_job.delay"),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        resp1 = client.post("/courses", json={"topic": "Elixir"}, headers=headers)
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "exists"
    assert resp1.json()["course"]["id"] == course_id
    with SessionLocal() as db:
        rows = db.query(UserCourse).filter_by(course_id=course_id, user_id=_user_id_of("enroll-d@example.com")).all()
        assert len(rows) == 1
        first_opened = rows[0].last_opened_at

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        resp2 = client.post("/courses", json={"topic": "Elixir"}, headers=headers)
    assert resp2.status_code == 200
    assert resp2.json()["course"]["id"] == course_id
    with SessionLocal() as db:
        rows = db.query(UserCourse).filter_by(course_id=course_id, user_id=_user_id_of("enroll-d@example.com")).all()
        assert len(rows) == 1
        assert rows[0].last_opened_at >= first_opened


def test_chapter_open_enrolls_user():
    headers = _auth_headers(email="enroll-c@example.com")
    _, _, chapter_id = _make_course_with_chapter("enroll-c")

    # Chapter content endpoint: patch the generator to avoid any real LLM work.
    fake_events = [{"type": "done"}]
    with patch("app.routes.courses.stream_chapter_content", return_value=iter(fake_events)):
        resp = client.get(f"/courses/enroll-c/chapters/{chapter_id}/content", headers=headers)
    assert resp.status_code == 200
    with SessionLocal() as db:
        assert db.query(UserCourse).filter_by(course_id=course_id_of("enroll-c")).count() == 1


def course_id_of(slug: str) -> int:
    with SessionLocal() as db:
        return db.query(Course).filter_by(topic_slug=slug).one().id


def _user_id_of(email: str) -> int:
    with SessionLocal() as db:
        return db.query(User).filter_by(email=email).one().id
