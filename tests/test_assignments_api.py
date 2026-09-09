from datetime import datetime, timezone
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.assignment import Assignment, AssignmentQuestion, AssignmentUserTopup
from app.models.user import User

client = TestClient(app)


def _auth_headers(email: str) -> dict:
    client.post("/auth/register", json={"email": email, "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": email, "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}


def _make_course_with_ready_chapter(slug: str, with_content: bool = True) -> tuple[int, int, int | None]:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
        db.add(chapter)
        db.commit()
        content_id = None
        if with_content:
            content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                      outline=[], created_at=now, updated_at=now)
            db.add(content)
            db.commit()
            db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H", kind="teaching",
                                        body_markdown="body", examples=[{"prompt": "p", "walkthrough": "w"}]))
            db.commit()
            content_id = content.id
        return chapter.id, content_id


def test_get_assignment_requires_auth():
    resp = client.get("/courses/nonexistent/chapters/1/assignment")
    assert resp.status_code == 401


def test_get_assignment_404_when_content_not_ready():
    headers = _auth_headers("asg-api-a@example.com")
    chapter_id, _ = _make_course_with_ready_chapter("asg-api-a", with_content=False)
    resp = client.get(f"/courses/asg-api-a/chapters/{chapter_id}/assignment", headers=headers)
    assert resp.status_code == 404


def test_get_assignment_dispatches_when_missing_then_returns_ready():
    headers = _auth_headers("asg-api-b@example.com")
    chapter_id, content_id = _make_course_with_ready_chapter("asg-api-b")

    def _fake_generate(cc_id):
        with SessionLocal() as db:
            assignment = Assignment(level="chapter", chapter_content_id=cc_id, scope="global",
                                    status="ready", created_at=datetime.now(timezone.utc),
                                    updated_at=datetime.now(timezone.utc))
            db.add(assignment)
            db.commit()
            db.add(AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q",
                                      options=["a", "b"], correct_answer="a", explanation="e",
                                      concept_tag="t", difficulty="easy"))
            db.commit()

    with patch("app.services.assignments.generate_chapter_assignment_task") as mock_task:
        mock_task.delay.side_effect = _fake_generate
        resp = client.get(f"/courses/asg-api-b/chapters/{chapter_id}/assignment", headers=headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert len(body["questions"]) == 1
    question = body["questions"][0]
    assert question == {"id": question["id"], "order": 0, "type": "mcq", "text": "q",
                        "options": ["a", "b"], "concept_tag": "t", "difficulty": "easy"}
    # the answer key never goes over the wire to the learner
    assert "correct_answer" not in question and "explanation" not in question
    mock_task.delay.assert_called_once_with(content_id)


def test_get_assignment_returns_generating_when_task_has_not_run_yet():
    headers = _auth_headers("asg-api-c@example.com")
    chapter_id, _ = _make_course_with_ready_chapter("asg-api-c")

    with patch("app.services.assignments.generate_chapter_assignment_task") as mock_task:
        resp = client.get(f"/courses/asg-api-c/chapters/{chapter_id}/assignment", headers=headers)

    assert resp.status_code == 200
    assert resp.json()["status"] == "generating"
    mock_task.delay.assert_called_once()


def test_get_assignment_retries_a_failed_assignment():
    headers = _auth_headers("asg-api-d@example.com")
    chapter_id, content_id = _make_course_with_ready_chapter("asg-api-d")
    with SessionLocal() as db:
        db.add(Assignment(level="chapter", chapter_content_id=content_id, scope="global", status="failed",
                          error="prior failure", created_at=datetime.now(timezone.utc),
                          updated_at=datetime.now(timezone.utc)))
        db.commit()

    with patch("app.services.assignments.generate_chapter_assignment_task") as mock_task:
        resp = client.get(f"/courses/asg-api-d/chapters/{chapter_id}/assignment", headers=headers)

    assert resp.status_code == 200
    mock_task.delay.assert_called_once_with(content_id)


def _make_module_via_api(slug: str, chapter_count: int = 1, all_ready: bool = True) -> tuple[int, int]:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        for i in range(chapter_count):
            chapter = Chapter(module_id=module.id, title=f"C{i}", objective="o", order=i)
            db.add(chapter)
            db.commit()
            content = ChapterContent(chapter_id=chapter.id, version=1, scope="global",
                                      status="ready" if all_ready else "generating",
                                      outline=[], created_at=now, updated_at=now)
            db.add(content)
            db.commit()
        return course.id, module.id


def test_post_module_assignment_409_when_a_chapter_not_ready():
    headers = _auth_headers("asg-api-mod-a@example.com")
    _, module_id = _make_module_via_api("asg-api-mod-a", chapter_count=1, all_ready=False)
    resp = client.post(f"/courses/asg-api-mod-a/modules/{module_id}/assignment", headers=headers)
    assert resp.status_code == 409


def test_post_module_assignment_dispatches_and_returns_ready():
    headers = _auth_headers("asg-api-mod-b@example.com")
    _, module_id = _make_module_via_api("asg-api-mod-b", chapter_count=1, all_ready=True)

    def _fake_generate(m_id):
        with SessionLocal() as db:
            assignment = Assignment(level="module", module_id=m_id, scope="global", status="ready",
                                    created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
            db.add(assignment)
            db.commit()
            db.add(AssignmentQuestion(assignment_id=assignment.id, order=0, type="true_false", text="q",
                                      options=None, correct_answer="true", explanation="e",
                                      concept_tag="t", difficulty="medium"))
            db.commit()

    with patch("app.services.assignments.generate_module_assignment_task") as mock_task:
        mock_task.delay.side_effect = _fake_generate
        resp = client.post(f"/courses/asg-api-mod-b/modules/{module_id}/assignment", headers=headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    question = body["questions"][0]
    assert question == {"id": question["id"], "order": 0, "type": "true_false", "text": "q",
                        "options": None, "concept_tag": "t", "difficulty": "medium"}
    assert "correct_answer" not in question and "explanation" not in question
    mock_task.delay.assert_called_once_with(module_id)


def test_get_module_assignment_retries_when_missing():
    headers = _auth_headers("asg-api-mod-c@example.com")
    _, module_id = _make_module_via_api("asg-api-mod-c", chapter_count=1, all_ready=True)

    with patch("app.services.assignments.generate_module_assignment_task") as mock_task:
        resp = client.get(f"/courses/asg-api-mod-c/modules/{module_id}/assignment", headers=headers)

    assert resp.status_code == 200
    assert resp.json()["status"] == "generating"
    mock_task.delay.assert_called_once_with(module_id)


def test_get_module_assignment_409_when_a_chapter_not_ready_and_nothing_generated_yet():
    headers = _auth_headers("asg-api-mod-d@example.com")
    _, module_id = _make_module_via_api("asg-api-mod-d", chapter_count=1, all_ready=False)
    resp = client.get(f"/courses/asg-api-mod-d/modules/{module_id}/assignment", headers=headers)
    assert resp.status_code == 409


def _make_ready_module_assignment(slug: str) -> tuple[int, int, int]:
    """Returns (module_id, assignment_id, base_question_id) for an
    already-"ready" module-level global assignment with one base question
    (user_id=None) — no chapters needed since these tests exercise the
    already-ready branch of get_module_assignment directly."""
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        assignment = Assignment(level="module", module_id=module.id, scope="global", status="ready",
                                created_at=now, updated_at=now)
        db.add(assignment)
        db.commit()
        question = AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="base q",
                                      options=["a", "b"], correct_answer="a", explanation="e",
                                      concept_tag="base", difficulty="easy")
        db.add(question)
        db.commit()
        return module.id, assignment.id, question.id


def test_get_module_assignment_dispatches_topup_once_per_user():
    headers = _auth_headers("asg-api-topup-a@example.com")
    slug = "asg-api-topup-a"
    module_id, assignment_id, _ = _make_ready_module_assignment(slug)
    with SessionLocal() as db:
        user_id = db.query(User).filter_by(email="asg-api-topup-a@example.com").one().id

    with patch("app.services.assignments.generate_module_topup_task") as mock_task:
        resp = client.get(f"/courses/{slug}/modules/{module_id}/assignment", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    mock_task.delay.assert_called_once_with(assignment_id, user_id)

    with SessionLocal() as db:
        topups = db.query(AssignmentUserTopup).filter_by(assignment_id=assignment_id, user_id=user_id).all()
        assert len(topups) == 1
        assert topups[0].status == "generating"

    # A second fetch by the SAME user must not dispatch again — the topup row
    # already exists.
    with patch("app.services.assignments.generate_module_topup_task") as mock_task_2:
        resp2 = client.get(f"/courses/{slug}/modules/{module_id}/assignment", headers=headers)
    assert resp2.status_code == 200
    mock_task_2.delay.assert_not_called()


def test_get_module_assignment_includes_only_own_ready_topup_questions():
    slug = "asg-api-topup-b"
    module_id, assignment_id, base_qid = _make_ready_module_assignment(slug)

    owner_headers = _auth_headers("asg-api-topup-b-owner@example.com")
    other_headers = _auth_headers("asg-api-topup-b-other@example.com")
    with SessionLocal() as db:
        owner_id = db.query(User).filter_by(email="asg-api-topup-b-owner@example.com").one().id
        now = datetime.now(timezone.utc)
        db.add(AssignmentUserTopup(assignment_id=assignment_id, user_id=owner_id, status="ready",
                                   created_at=now, updated_at=now))
        db.commit()
        topup_q = AssignmentQuestion(assignment_id=assignment_id, order=1, type="mcq", text="owner topup q",
                                     options=["a", "b"], correct_answer="a", explanation="e",
                                     concept_tag="owner-topup", difficulty="easy", user_id=owner_id)
        db.add(topup_q)
        db.commit()
        topup_qid = topup_q.id

    # The owner's topup already exists and is "ready" — no dispatch expected.
    with patch("app.services.assignments.generate_module_topup_task") as mock_task:
        resp = client.get(f"/courses/{slug}/modules/{module_id}/assignment", headers=owner_headers)
    assert resp.status_code == 200
    mock_task.delay.assert_not_called()
    owner_question_ids = {q["id"] for q in resp.json()["questions"]}
    assert owner_question_ids == {base_qid, topup_qid}

    # A DIFFERENT user fetching the same module assignment must not see the
    # owner's topup question (their own topup dispatch is fire-and-forget and
    # not ready yet, so only the shared base question shows).
    with patch("app.services.assignments.generate_module_topup_task"):
        other_resp = client.get(f"/courses/{slug}/modules/{module_id}/assignment", headers=other_headers)
    assert other_resp.status_code == 200
    other_question_ids = {q["id"] for q in other_resp.json()["questions"]}
    assert other_question_ids == {base_qid}
    assert topup_qid not in other_question_ids
