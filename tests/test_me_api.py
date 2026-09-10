from datetime import datetime, timezone
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter, Concept
from app.models.enrollment import UserCourse
from app.models.chapter_content import ChapterContent
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.user import User
from app.services import streaks

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
    slugs = {r["topic_slug"] for r in body["items"]}
    assert slugs == {"me-a-1", "me-a-2"}
    row = next(r for r in body["items"] if r["topic_slug"] == "me-a-1")
    assert row["module_count"] == 1
    assert row["chapter_count"] == 1
    assert row["content_ready"] is False


def test_me_courses_paginated_envelope():
    headers = _auth_for("me-pg@example.com")
    user_id = _get_user_id("me-pg@example.com")
    _enroll(user_id, _make_course("me-pg-1"))
    _enroll(user_id, _make_course("me-pg-2"))
    _enroll(user_id, _make_course("me-pg-3"))

    resp = client.get("/me/courses?page=1&limit=2", headers=headers)
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["page"] == 1
    assert body["limit"] == 2
    assert body["total_pages"] == 2


def test_me_courses_search_filters_by_title():
    headers = _auth_for("me-srch@example.com")
    user_id = _get_user_id("me-srch@example.com")
    _enroll(user_id, _make_course("me-srch-rust"))
    _enroll(user_id, _make_course("me-srch-elixir"))

    resp = client.get("/me/courses?search=rust", headers=headers)
    slugs = {r["topic_slug"] for r in resp.json()["items"]}
    assert slugs == {"me-srch-rust"}


def test_me_courses_status_filter():
    headers = _auth_for("me-status@example.com")
    user_id = _get_user_id("me-status@example.com")
    c1 = _make_course("me-status-1")
    c2 = _make_course("me-status-2")
    _enroll(user_id, c1)
    _enroll(user_id, c2)
    with SessionLocal() as db:
        row = db.query(UserCourse).filter_by(user_id=user_id, course_id=c2).one()
        row.status = "completed"
        db.commit()

    resp = client.get("/me/courses?status=completed", headers=headers)
    slugs = {r["topic_slug"] for r in resp.json()["items"]}
    assert slugs == {"me-status-2"}


def test_me_courses_sort_by_name_asc():
    headers = _auth_for("me-sort@example.com")
    user_id = _get_user_id("me-sort@example.com")
    _enroll(user_id, _make_course("me-sort-zebra"))
    _enroll(user_id, _make_course("me-sort-alpha"))

    resp = client.get("/me/courses?sort=name&order=asc", headers=headers)
    slugs = [r["topic_slug"] for r in resp.json()["items"]]
    assert slugs.index("me-sort-alpha") < slugs.index("me-sort-zebra")


def test_me_courses_content_ready_true():
    headers = _auth_for("me-b@example.com")
    user_id = _get_user_id("me-b@example.com")
    c = _make_course("me-b-1", content_ready=True)
    _enroll(user_id, c)
    resp = client.get("/me/courses", headers=headers)
    row = next(r for r in resp.json()["items"] if r["topic_slug"] == "me-b-1")
    assert row["content_ready"] is True


def test_me_courses_content_ready_two_chapters_all_ready():
    headers = _auth_for("me-f@example.com")
    user_id = _get_user_id("me-f@example.com")
    c = _make_course("me-f-1", content_ready=True, num_chapters=2)
    _enroll(user_id, c)
    resp = client.get("/me/courses", headers=headers)
    row = next(r for r in resp.json()["items"] if r["topic_slug"] == "me-f-1")
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
        assert chapter is not None
        db.add(ChapterContent(chapter_id=chapter.id, version=1, scope="global",
                              status="ready", outline=[],
                              created_at=now, updated_at=now))
        db.commit()
    _enroll(user_id, c)
    resp = client.get("/me/courses", headers=headers)
    row = next(r for r in resp.json()["items"] if r["topic_slug"] == "me-g-1")
    assert row["chapter_count"] == 2
    assert row["content_ready"] is False


def test_me_courses_content_ready_no_chapters_is_false():
    headers = _auth_for("me-h@example.com")
    user_id = _get_user_id("me-h@example.com")
    c = _make_course("me-h-1", with_chapter=False)
    _enroll(user_id, c)
    resp = client.get("/me/courses", headers=headers)
    row = next(r for r in resp.json()["items"] if r["topic_slug"] == "me-h-1")
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


def test_dashboard_includes_streak_and_mastery_counts():
    headers = _auth_for("me-i@example.com")
    user_id = _get_user_id("me-i@example.com")

    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug="me-i-1", topic_raw="me-i-1", topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course); db.flush()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module); db.flush()
        weak_chapter = Chapter(module_id=module.id, title="Weak", objective="o", order=1)
        db.add(weak_chapter); db.flush()
        weak_content = ChapterContent(chapter_id=weak_chapter.id, version=1, scope="global", status="ready",
                                       outline=[], remediation_target_tags=["weak-tag"],
                                       created_at=now, updated_at=now)
        db.add(weak_content); db.flush()
        weak_assignment = Assignment(level="chapter", chapter_content_id=weak_content.id, scope="global",
                                      status="ready", created_at=now, updated_at=now)
        db.add(weak_assignment); db.flush()
        db.add(Concept(course_id=course.id, name="weak-tag", chapter_id=weak_chapter.id))

        strong_chapter = Chapter(module_id=module.id, title="Strong", objective="o", order=2)
        db.add(strong_chapter); db.flush()
        strong_content = ChapterContent(chapter_id=strong_chapter.id, version=1, scope="global", status="ready",
                                         outline=[], created_at=now, updated_at=now)
        db.add(strong_content); db.flush()
        strong_assignment = Assignment(level="chapter", chapter_content_id=strong_content.id, scope="global",
                                        status="ready", created_at=now, updated_at=now)
        db.add(strong_assignment); db.flush()
        db.add(Concept(course_id=course.id, name="strong-tag", chapter_id=strong_chapter.id))
        db.flush()

        db.add(AssignmentAttempt(assignment_id=weak_assignment.id, user_id=user_id, status="graded",
                                  overall_score=0.2, created_at=now, updated_at=now))
        db.add(AssignmentAttempt(assignment_id=strong_assignment.id, user_id=user_id, status="graded",
                                  overall_score=0.9, created_at=now, updated_at=now))
        db.add(UserCourse(user_id=user_id, course_id=course.id, enrolled_at=now, last_opened_at=now))
        db.commit()
        course_id = course.id

    with SessionLocal() as db:
        streaks.record_activity(db, user_id, datetime.now(timezone.utc))
        db.commit()

    resp = client.get("/me/dashboard", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["streak"]["current"] == 1
    row = next(r for r in body["in_progress"] if r["id"] == course_id)
    assert row["weak_concept_count"] == 1
    assert row["strong_concept_count"] == 1


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
