from datetime import datetime, timezone
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt, AssignmentAnswer

client = TestClient(app)


def _auth_headers(email: str) -> dict:
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    tokens = client.post("/auth/login", json={"email": email, "password": "pw123456"}).json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def _now():
    return datetime.now(timezone.utc)


def _make_ready_assignment(slug: str, question_count: int = 2, status: str = "ready") -> tuple[str, int, list[int]]:
    """Returns (course_slug, assignment_id, question_ids)."""
    with SessionLocal() as db:
        course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
        db.add(chapter)
        db.commit()
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                  outline=[], created_at=_now(), updated_at=_now())
        db.add(content)
        db.commit()
        assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                                 status=status, created_at=_now(), updated_at=_now())
        db.add(assignment)
        db.commit()
        question_ids = []
        for i in range(question_count):
            q = AssignmentQuestion(assignment_id=assignment.id, order=i, type="mcq", text=f"q{i}",
                                    options=["a", "b"], correct_answer="a", explanation="e",
                                    concept_tag="t", difficulty="easy")
            db.add(q)
            db.commit()
            question_ids.append(q.id)
        return course.topic_slug, assignment.id, question_ids


def test_submit_requires_auth():
    resp = client.post("/courses/nonexistent/assignments/1/attempts", json={"answers": []})
    assert resp.status_code == 401


def test_submit_404_when_assignment_not_ready():
    headers = _auth_headers("att-api-a@example.com")
    slug, assignment_id, question_ids = _make_ready_assignment("att-api-a", status="generating")
    resp = client.post(
        f"/courses/{slug}/assignments/{assignment_id}/attempts",
        json={"answers": [{"question_id": qid, "answer": "a"} for qid in question_ids]},
        headers=headers,
    )
    assert resp.status_code == 404


def test_submit_400_when_answers_dont_cover_question_set():
    headers = _auth_headers("att-api-b@example.com")
    slug, assignment_id, question_ids = _make_ready_assignment("att-api-b", question_count=2)
    resp = client.post(
        f"/courses/{slug}/assignments/{assignment_id}/attempts",
        json={"answers": [{"question_id": question_ids[0], "answer": "a"}]},  # missing the 2nd question
        headers=headers,
    )
    assert resp.status_code == 400


def test_submit_400_when_answer_references_unknown_question():
    headers = _auth_headers("att-api-c@example.com")
    slug, assignment_id, question_ids = _make_ready_assignment("att-api-c", question_count=1)
    resp = client.post(
        f"/courses/{slug}/assignments/{assignment_id}/attempts",
        json={"answers": [{"question_id": question_ids[0], "answer": "a"}, {"question_id": 999999, "answer": "b"}]},
        headers=headers,
    )
    assert resp.status_code == 400


def test_submit_creates_attempt_and_dispatches_grading():
    headers = _auth_headers("att-api-d@example.com")
    slug, assignment_id, question_ids = _make_ready_assignment("att-api-d", question_count=2)

    with patch("app.routes.courses.grade_assignment_attempt_task") as mock_task:
        resp = client.post(
            f"/courses/{slug}/assignments/{assignment_id}/attempts",
            json={"answers": [{"question_id": qid, "answer": "a"} for qid in question_ids]},
            headers=headers,
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "grading"
    attempt_id = body["attempt_id"]
    mock_task.delay.assert_called_once_with(attempt_id)

    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt.status == "grading"
        answers = db.query(AssignmentAnswer).filter_by(attempt_id=attempt_id).all()
        assert len(answers) == 2
        assert all(a.concept_tag == "t" for a in answers)  # denormalized at submit time


def test_submit_allows_a_second_attempt_on_the_same_assignment():
    headers = _auth_headers("att-api-e@example.com")
    slug, assignment_id, question_ids = _make_ready_assignment("att-api-e", question_count=1)

    with patch("app.routes.courses.grade_assignment_attempt_task"):
        resp1 = client.post(
            f"/courses/{slug}/assignments/{assignment_id}/attempts",
            json={"answers": [{"question_id": question_ids[0], "answer": "a"}]},
            headers=headers,
        )
        resp2 = client.post(
            f"/courses/{slug}/assignments/{assignment_id}/attempts",
            json={"answers": [{"question_id": question_ids[0], "answer": "b"}]},
            headers=headers,
        )

    assert resp1.status_code == 202
    assert resp2.status_code == 202
    assert resp1.json()["attempt_id"] != resp2.json()["attempt_id"]


def test_submit_400_when_duplicate_question_id():
    headers = _auth_headers("att-api-f@example.com")
    slug, assignment_id, question_ids = _make_ready_assignment("att-api-f", question_count=1)
    resp = client.post(
        f"/courses/{slug}/assignments/{assignment_id}/attempts",
        json={"answers": [{"question_id": question_ids[0], "answer": "a"}, {"question_id": question_ids[0], "answer": "b"}]},
        headers=headers,
    )
    assert resp.status_code == 400
