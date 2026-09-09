from datetime import datetime, timezone
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.assignment import Assignment, AssignmentQuestion, AssignmentUserTopup
from app.models.user import User
from app.agents.assignment.generate import generate_module_topup, TOPUP_QUESTION_COUNT
from app.agents.assignment.nodes.generate_section_questions import QuestionDraft

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


def _make_ready_module_assignment_with_weak_chapter(slug: str) -> tuple[int, int, int]:
    """Like `_make_ready_module_assignment`, but the module's one chapter
    carries `remediation_target_tags` and no learner has a passing attempt
    for it, so `mastery.get_weak_concept_tags` returns a non-empty set for
    any user on this course — lets a topup dispatch reach the LLM-call
    branch instead of short-circuiting to "skipped". Returns (module_id,
    assignment_id, base_question_id)."""
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
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                  outline=[], remediation_target_tags=["weak-tag"],
                                  created_at=now, updated_at=now)
        db.add(content)
        db.commit()
        db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H", kind="teaching",
                                    body_markdown="body", examples=[{"prompt": "p", "walkthrough": "w"}]))
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


def _make_ready_module_assignment_with_chapters(slug: str) -> tuple[int, int, int]:
    """Like `_make_ready_module_assignment` but with one ready chapter under
    the module, so `create_module_assignment` (POST)'s always-on chapter
    readiness gate (409 if not all chapters have ready content) doesn't
    trip before the topup logic these tests exercise is even reached —
    unlike `get_module_assignment`, which only gates on readiness when a
    dispatch is actually needed. Returns (module_id, assignment_id,
    base_question_id)."""
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
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                  outline=[], created_at=now, updated_at=now)
        db.add(content)
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


def _weak_draft(tag: str) -> QuestionDraft:
    return QuestionDraft(type="mcq", text=f"q-{tag}", options=["a", "b"], correct_answer="a",
                        explanation="e", concept_tag=tag, difficulty="easy")


def test_get_module_assignment_dispatches_topup_once_per_user():
    """C1 regression: the route must NOT pre-create the AssignmentUserTopup
    row before dispatching — a prior version did, which made every real
    `generate_module_topup` run see `created=False` and a fresh
    status="generating", conclude a healthy run was already in flight, and
    no-op forever (a permanent stranded row, the feature silently doing
    nothing in production). The mock here has no side effect, so if the
    route were still creating the row itself, a topup row WOULD exist after
    this fetch even though nothing actually ran it — asserting no row exists
    yet is exactly what would have caught that regression."""
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
        # The task is mocked with no side effect (never actually runs), so
        # the route dispatching it must not itself have created the row —
        # row creation belongs entirely to generate.py's own get-or-create.
        topups = db.query(AssignmentUserTopup).filter_by(assignment_id=assignment_id, user_id=user_id).all()
        assert len(topups) == 0

    # A second fetch by the SAME user, with the (real, unmocked) task's
    # get-or-create actually running this time, creates exactly one row and
    # dispatches exactly once.
    def _fake_generate(a_id, u_id):
        with SessionLocal() as db:
            generate_module_topup(a_id, u_id, db)

    with patch("app.services.assignments.generate_module_topup_task") as mock_task_2:
        mock_task_2.delay.side_effect = _fake_generate
        resp2 = client.get(f"/courses/{slug}/modules/{module_id}/assignment", headers=headers)
    assert resp2.status_code == 200
    mock_task_2.delay.assert_called_once_with(assignment_id, user_id)
    with SessionLocal() as db:
        topups = db.query(AssignmentUserTopup).filter_by(assignment_id=assignment_id, user_id=user_id).all()
        assert len(topups) == 1
        assert topups[0].status == "skipped"  # no chapters/weak concepts set up for this module

    # A third fetch must not dispatch again — the (now "skipped", terminal)
    # topup row already exists.
    with patch("app.services.assignments.generate_module_topup_task") as mock_task_3:
        resp3 = client.get(f"/courses/{slug}/modules/{module_id}/assignment", headers=headers)
    assert resp3.status_code == 200
    mock_task_3.delay.assert_not_called()


def test_get_module_assignment_topup_dispatch_actually_generates_and_reaches_ready():
    """C1: the test class the final review said was missing — every existing
    test stopped at a mock boundary (asserting `.delay` was called) that hid
    the no-op bug. This test drives BOTH the route's dispatch decision AND
    the actual task body (`generate_module_topup`) together, mocking only
    the LLM call, and asserts the topup genuinely reaches "ready" with its
    questions persisted and visible in the response — not stranded at
    "generating"."""
    headers = _auth_headers("asg-api-topup-c1@example.com")
    slug = "asg-api-topup-c1"
    module_id, assignment_id, base_qid = _make_ready_module_assignment_with_weak_chapter(slug)
    with SessionLocal() as db:
        user_id = db.query(User).filter_by(email="asg-api-topup-c1@example.com").one().id

    drafts = [_weak_draft(f"weak-{i}") for i in range(TOPUP_QUESTION_COUNT)]

    def _run_real_task(a_id, u_id):
        with SessionLocal() as db:
            with patch("app.agents.assignment.generate.generate_weak_concept_questions", return_value=drafts):
                generate_module_topup(a_id, u_id, db)

    with patch("app.services.assignments.generate_module_topup_task") as mock_task:
        mock_task.delay.side_effect = _run_real_task
        resp = client.get(f"/courses/{slug}/modules/{module_id}/assignment", headers=headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    question_ids = {q["id"] for q in body["questions"]}
    assert base_qid in question_ids
    assert len(question_ids) == 1 + TOPUP_QUESTION_COUNT  # base + all generated topup questions

    with SessionLocal() as db:
        topup = db.query(AssignmentUserTopup).filter_by(assignment_id=assignment_id, user_id=user_id).one()
        assert topup.status == "ready"  # not stranded at "generating"
        questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment_id, user_id=user_id).all()
        assert len(questions) == TOPUP_QUESTION_COUNT


def test_get_module_assignment_redispatches_a_failed_topup():
    """I1: a "failed" topup must be re-dispatched on a subsequent fetch,
    mirroring the `assignment is None or assignment.status == "failed"`
    convention already used for the base assignment."""
    headers = _auth_headers("asg-api-topup-i1@example.com")
    slug = "asg-api-topup-i1"
    module_id, assignment_id, _ = _make_ready_module_assignment(slug)
    with SessionLocal() as db:
        user_id = db.query(User).filter_by(email="asg-api-topup-i1@example.com").one().id
        now = datetime.now(timezone.utc)
        db.add(AssignmentUserTopup(assignment_id=assignment_id, user_id=user_id, status="failed",
                                   error="prior failure", created_at=now, updated_at=now))
        db.commit()

    with patch("app.services.assignments.generate_module_topup_task") as mock_task:
        resp = client.get(f"/courses/{slug}/modules/{module_id}/assignment", headers=headers)

    assert resp.status_code == 200
    mock_task.delay.assert_called_once_with(assignment_id, user_id)


def test_post_module_assignment_redispatches_a_failed_topup():
    """I1, POST path."""
    headers = _auth_headers("asg-api-topup-i1-post@example.com")
    slug = "asg-api-topup-i1-post"
    module_id, assignment_id, _ = _make_ready_module_assignment_with_chapters(slug)
    with SessionLocal() as db:
        user_id = db.query(User).filter_by(email="asg-api-topup-i1-post@example.com").one().id
        now = datetime.now(timezone.utc)
        db.add(AssignmentUserTopup(assignment_id=assignment_id, user_id=user_id, status="failed",
                                   error="prior failure", created_at=now, updated_at=now))
        db.commit()

    with patch("app.services.assignments.generate_module_topup_task") as mock_task:
        resp = client.post(f"/courses/{slug}/modules/{module_id}/assignment", headers=headers)

    assert resp.status_code == 200
    mock_task.delay.assert_called_once_with(assignment_id, user_id)


def test_post_module_assignment_does_not_leak_other_learners_topup_questions():
    """C2 regression: `create_module_assignment` (POST) used to call
    `_serialize_assignment(assignment, db)` with no `user_id` at all — every
    learner got every OTHER learner's topup questions, and the returned set
    didn't match what `submit_attempt` would accept for that learner."""
    slug = "asg-api-topup-c2"
    module_id, assignment_id, base_qid = _make_ready_module_assignment_with_chapters(slug)

    owner_headers = _auth_headers("asg-api-topup-c2-owner@example.com")
    other_headers = _auth_headers("asg-api-topup-c2-other@example.com")
    with SessionLocal() as db:
        owner_id = db.query(User).filter_by(email="asg-api-topup-c2-owner@example.com").one().id
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

    # A DIFFERENT learner POSTing to the same module assignment must see only
    # the shared base question — never the owner's topup question.
    with patch("app.services.assignments.generate_module_topup_task"):
        other_resp = client.post(f"/courses/{slug}/modules/{module_id}/assignment", headers=other_headers)
    assert other_resp.status_code == 200
    other_question_ids = {q["id"] for q in other_resp.json()["questions"]}
    assert other_question_ids == {base_qid}
    assert topup_qid not in other_question_ids

    # The owner's own POST must include their own topup (not the leaked
    # every-learner set the pre-fix version returned).
    with patch("app.services.assignments.generate_module_topup_task"):
        owner_resp = client.post(f"/courses/{slug}/modules/{module_id}/assignment", headers=owner_headers)
    assert owner_resp.status_code == 200
    owner_question_ids = {q["id"] for q in owner_resp.json()["questions"]}
    assert owner_question_ids == {base_qid, topup_qid}


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
