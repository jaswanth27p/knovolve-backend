from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.enrollment import UserCourse
from app.models.chapter_content import ChapterContent
from app.models.user import User

client = TestClient(app)


def _register(email: str) -> dict:
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": email, "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}


def _make_course(slug: str, with_chapter: bool = True, content_ready: bool = False,
                 num_chapters: int = 1):
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        count = num_chapters if with_chapter else 0
        for i in range(count):
            chapter = Chapter(module_id=module.id, title="C", objective="o", order=i + 1)
            db.add(chapter)
            db.commit()
            if content_ready:
                db.add(ChapterContent(chapter_id=chapter.id, version=1, scope="global",
                                      status="ready", outline=[],
                                      created_at=now, updated_at=now))
                db.commit()
        return course.id


def _enroll(user_id: int, course_id: int):
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        db.add(UserCourse(user_id=user_id, course_id=course_id, enrolled_at=now, last_opened_at=now))
        db.commit()


def _get_user_id(email: str) -> int:
    with SessionLocal() as db:
        return db.query(User).filter_by(email=email).one().id


def test_me_courses_requires_auth():
    resp = client.get("/me/courses")
    assert resp.status_code == 401


def test_me_courses_returns_only_own():
    headers = _auth_for("me-a@example.com")
    user_id = _get_user_id("me-a@example.com")
    c1 = _make_course("me-a-1")
    c2 = _make_course("me-a-2")
    _enroll(user_id, c1)
    _enroll(user_id, c2)

    resp = client.get("/me/courses", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    slugs = {r["topic_slug"] for r in body}
    assert slugs == {"me-a-1", "me-a-2"}
    row = next(r for r in body if r["topic_slug"] == "me-a-1")
    assert row["module_count"] == 1
    assert row["chapter_count"] == 1
    assert row["content_ready"] is False


def test_me_courses_content_ready_true():
    headers = _auth_for("me-b@example.com")
    user_id = _get_user_id("me-b@example.com")
    c = _make_course("me-b-1", content_ready=True)
    _enroll(user_id, c)
    resp = client.get("/me/courses", headers=headers)
    row = next(r for r in resp.json() if r["topic_slug"] == "me-b-1")
    assert row["content_ready"] is True


def test_me_courses_content_ready_two_chapters_all_ready():
    headers = _auth_for("me-f@example.com")
    user_id = _get_user_id("me-f@example.com")
    c = _make_course("me-f-1", content_ready=True, num_chapters=2)
    _enroll(user_id, c)
    resp = client.get("/me/courses", headers=headers)
    row = next(r for r in resp.json() if r["topic_slug"] == "me-f-1")
    assert row["chapter_count"] == 2
    assert row["content_ready"] is True


def test_me_courses_content_ready_partial_is_false():
    headers = _auth_for("me-g@example.com")
    user_id = _get_user_id("me-g@example.com")
    c = _make_course("me-g-1", num_chapters=2)
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = db.query(Course).filter_by(topic_slug="me-g-1").one()
        module = db.query(Module).filter_by(course_id=course.id).one()
        chapter = db.query(Chapter).filter_by(module_id=module.id).order_by(Chapter.order).first()
        db.add(ChapterContent(chapter_id=chapter.id, version=1, scope="global",
                              status="ready", outline=[],
                              created_at=now, updated_at=now))
        db.commit()
    _enroll(user_id, c)
    resp = client.get("/me/courses", headers=headers)
    row = next(r for r in resp.json() if r["topic_slug"] == "me-g-1")
    assert row["chapter_count"] == 2
    assert row["content_ready"] is False


def test_me_courses_content_ready_no_chapters_is_false():
    headers = _auth_for("me-h@example.com")
    user_id = _get_user_id("me-h@example.com")
    c = _make_course("me-h-1", with_chapter=False)
    _enroll(user_id, c)
    resp = client.get("/me/courses", headers=headers)
    row = next(r for r in resp.json() if r["topic_slug"] == "me-h-1")
    assert row["chapter_count"] == 0
    assert row["content_ready"] is False


def test_me_dashboard_counts():
    headers = _auth_for("me-c@example.com")
    user_id = _get_user_id("me-c@example.com")
    _enroll(user_id, _make_course("me-c-1", content_ready=True))
    _enroll(user_id, _make_course("me-c-2"))

    resp = client.get("/me/dashboard", headers=headers)
    body = resp.json()
    assert body["in_progress_count"] == 2
    assert body["completed_count"] == 0
    assert body["total_count"] == 2


def test_delete_me_course_un_tracks_only():
    headers_a = _auth_for("me-d-a@example.com")
    headers_b = _auth_for("me-d-b@example.com")
    uid_a = _get_user_id("me-d-a@example.com")
    uid_b = _get_user_id("me-d-b@example.com")
    c = _make_course("me-d-1")
    _enroll(uid_a, c)
    _enroll(uid_b, c)

    resp = client.delete(f"/me/courses/{c}", headers=headers_a)
    assert resp.status_code == 204
    with SessionLocal() as db:
        assert db.query(UserCourse).filter_by(user_id=uid_a, course_id=c).count() == 0
        assert db.query(UserCourse).filter_by(user_id=uid_b, course_id=c).count() == 1


def test_delete_me_course_404_when_not_tracked():
    headers = _auth_for("me-e@example.com")
    c = _make_course("me-e-1")
    resp = client.delete(f"/me/courses/{c}", headers=headers)
    assert resp.status_code == 404


def _auth_for(email: str) -> dict:
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": email, "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}
