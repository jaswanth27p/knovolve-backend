from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt, AssignmentAnswer
from app.models.user import User
from app.models.learner_streak import LearnerStreak
from app.agents.evaluation.nodes.grade_assignment_answers import AnswerGrade, GradingResponse
from app.agents.evaluation.grade import grade_assignment_attempt
from app.tasks.evaluation_tasks import grade_assignment_attempt_task


def _now():
    return datetime.now(timezone.utc)


def _make_assignment_with_questions(db, slug: str, questions: list[dict]) -> int:
    """questions: list of {"type", "correct_answer", "concept_tag"}. Returns assignment_id."""
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
    db.add(course)
    db.flush()
    module = Module(course_id=course.id, title="M", objective="o", order=1)
    db.add(module)
    db.flush()
    chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
    db.add(chapter)
    db.flush()
    content = ChapterContent(chapter_id=chapter.id, version=1, scope="global",
                              status="ready", outline=[], created_at=_now(), updated_at=_now())
    db.add(content)
    db.flush()
    assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                             status="ready", created_at=_now(), updated_at=_now())
    db.add(assignment)
    db.flush()
    for i, q in enumerate(questions):
        db.add(AssignmentQuestion(
            assignment_id=assignment.id, order=i, type=q["type"], text=f"q{i}",
            options=["a", "b"] if q["type"] == "mcq" else None,
            correct_answer=q["correct_answer"], explanation="e",
            concept_tag=q["concept_tag"], difficulty="easy",
        ))
    db.flush()
    return assignment.id


def _make_attempt(db, assignment_id: int, answers: dict[int, str]) -> int:
    """answers: {question_order: user_answer}. Returns attempt_id."""
    user = User(email=f"grader-{assignment_id}@example.com", password_hash="x")
    db.add(user)
    db.flush()
    attempt = AssignmentAttempt(assignment_id=assignment_id, user_id=user.id, status="grading",
                                 created_at=_now(), updated_at=_now())
    db.add(attempt)
    db.flush()
    questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment_id).order_by(AssignmentQuestion.order).all()
    for order, user_answer in answers.items():
        q = questions[order]
        db.add(AssignmentAnswer(attempt_id=attempt.id, question_id=q.id, concept_tag=q.concept_tag,
                                 user_answer=user_answer))
    db.commit()
    return attempt.id


def _chapter_id_for_assignment(db, assignment_id: int) -> int:
    assignment = db.get(Assignment, assignment_id)
    assert assignment is not None and assignment.chapter_content_id is not None
    content = db.get(ChapterContent, assignment.chapter_content_id)
    assert content is not None
    return content.chapter_id


def test_grades_mcq_and_true_false_deterministically_llm_called_for_verdict_only():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-a", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
            {"type": "true_false", "correct_answer": "true", "concept_tag": "t2"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a", 1: "false"})

    fake_response = GradingResponse(grades=[], remediation_concept_tags=[],
                                     verdict_reasoning="Both deterministic answers graded; nothing to remediate.")
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response) as mock_grader:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    # The LLM node is still called (for the holistic verdict) even though
    # there are zero free-text questions — spec 03 requires this.
    mock_grader.assert_called_once()
    free_text_arg, known_arg = mock_grader.call_args.args
    assert free_text_arg == []
    assert len(known_arg) == 2  # both mcq and true_false passed as known-answer context

    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        assert attempt.status == "graded"
        assert attempt.overall_score == 0.5
        assert attempt.verdict_reasoning == "Both deterministic answers graded; nothing to remediate."
        answers = db.query(AssignmentAnswer).filter_by(attempt_id=attempt_id).order_by(AssignmentAnswer.question_id).all()
        assert answers[0].is_correct is True
        assert answers[1].is_correct is False
        assert answers[0].graded_at is not None
        assert answers[0].misconception_tag is None
        assert answers[1].misconception_tag is None


def test_exact_match_is_case_insensitive_and_trimmed():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-b", [
            {"type": "true_false", "correct_answer": "True", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "  true  "})

    fake_response = GradingResponse(grades=[], remediation_concept_tags=[], verdict_reasoning="ok")
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response):
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    with SessionLocal() as db:
        answer = db.query(AssignmentAnswer).filter_by(attempt_id=attempt_id).one()
        assert answer.is_correct is True


def test_free_text_answers_batched_into_one_llm_call():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-c", [
            {"type": "free_text", "correct_answer": "X is Y", "concept_tag": "t1"},
            {"type": "free_text", "correct_answer": "Z is W", "concept_tag": "t2"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "X is Y basically", 1: "no idea"})
        questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment_id).order_by(AssignmentQuestion.order).all()
        q0_id, q1_id = questions[0].id, questions[1].id

    fake_response = GradingResponse(
        grades=[
            AnswerGrade(question_id=q0_id, is_correct=True, feedback="Correct."),
            AnswerGrade(question_id=q1_id, is_correct=False, feedback="Incorrect.", misconception_tag="skipped-the-question"),
        ],
        remediation_concept_tags=[], verdict_reasoning="Partially correct.",
    )
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response) as mock_grader:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    mock_grader.assert_called_once()
    free_text_arg, known_arg = mock_grader.call_args.args
    assert len(free_text_arg) == 2  # both items in one batched call
    assert known_arg == []
    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        assert attempt.overall_score == 0.5
        assert attempt.verdict_reasoning == "Partially correct."
        answers = {a.question_id: a for a in db.query(AssignmentAnswer).filter_by(attempt_id=attempt_id).all()}
        assert answers[q0_id].is_correct is True
        assert answers[q0_id].feedback == "Correct."
        assert answers[q0_id].misconception_tag is None
        assert answers[q1_id].is_correct is False
        assert answers[q1_id].misconception_tag == "skipped-the-question"


def test_mixed_mcq_and_free_text_in_one_attempt():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-d", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
            {"type": "free_text", "correct_answer": "X is Y", "concept_tag": "t2"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a", 1: "X is Y"})
        questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment_id).order_by(AssignmentQuestion.order).all()
        free_text_question_id = questions[1].id

    fake_response = GradingResponse(
        grades=[AnswerGrade(question_id=free_text_question_id, is_correct=True, feedback="Correct.")],
        remediation_concept_tags=[], verdict_reasoning="All good.",
    )
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response) as mock_grader:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    # Only the free_text item is sent for grading — the mcq item is passed as known-answer context, not re-graded.
    free_text_arg, known_arg = mock_grader.call_args.args
    assert len(free_text_arg) == 1
    assert free_text_arg[0].question_id == free_text_question_id
    assert len(known_arg) == 1
    assert known_arg[0].is_correct is True
    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        assert attempt.overall_score == 1.0


def test_failure_marks_attempt_failed_with_no_answers_graded():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-e", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
            {"type": "free_text", "correct_answer": "X is Y", "concept_tag": "t2"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a", 1: "X is Y"})

    with patch("app.agents.evaluation.grade.grade_assignment_answers", side_effect=RuntimeError("llm down")):
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        assert attempt.status == "failed"
        assert attempt.error == "llm down"
        assert attempt.overall_score is None
        assert attempt.verdict_reasoning is None
        # The mcq answer was graded in-session before the LLM call failed —
        # rollback must undo it too. All-or-nothing on the graded fields.
        answers = db.query(AssignmentAnswer).filter_by(attempt_id=attempt_id).all()
        assert all(a.is_correct is None for a in answers)
        assert all(a.graded_at is None for a in answers)


def test_partial_llm_grades_marks_attempt_failed_with_no_answers_graded():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-h", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
            {"type": "free_text", "correct_answer": "X is Y", "concept_tag": "t2"},
            {"type": "free_text", "correct_answer": "Z is W", "concept_tag": "t3"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a", 1: "X is Y", 2: "no idea"})
        questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment_id).order_by(AssignmentQuestion.order).all()
        q1_id, q2_id = questions[1].id, questions[2].id

    # Only q1 gets a grade back; q2's question_id is missing from the LLM response.
    partial_response = GradingResponse(
        grades=[AnswerGrade(question_id=q1_id, is_correct=True, feedback="Correct.")],
        remediation_concept_tags=[], verdict_reasoning="n/a",
    )
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=partial_response):
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        assert attempt.status == "failed"
        assert attempt.error is not None
        assert str(q2_id) in attempt.error
        assert attempt.overall_score is None
        # All-or-nothing: even the mcq answer graded before the check must be rolled back.
        answers = db.query(AssignmentAnswer).filter_by(attempt_id=attempt_id).all()
        assert all(a.is_correct is None for a in answers)
        assert all(a.graded_at is None for a in answers)


def test_idempotent_when_not_in_grading_status():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-f", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a"})
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        attempt.status = "graded"
        attempt.overall_score = 1.0
        db.commit()

    with patch("app.agents.evaluation.grade.grade_assignment_answers") as mock_grader:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)  # already graded — must no-op

    mock_grader.assert_not_called()


def test_task_wrapper_invokes_grading():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-g", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a"})

    with patch("app.tasks.evaluation_tasks.grade_assignment_attempt") as mock_grade:
        grade_assignment_attempt_task(attempt_id)  # pyright: ignore[reportCallIssue]

    mock_grade.assert_called_once()
    assert mock_grade.call_args.args[0] == attempt_id


def test_remediation_dispatched_for_chapter_level_attempt_with_remediation_tags():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-remediate-a", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "wrong"})
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        user_id = attempt.user_id
        chapter_id = _chapter_id_for_assignment(db, assignment_id)

    fake_response = GradingResponse(grades=[], remediation_concept_tags=["t1"], verdict_reasoning="Missed t1.")
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response), \
         patch("app.agents.evaluation.grade.remediate_chapter_task") as mock_remediate_task:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    mock_remediate_task.delay.assert_called_once_with(chapter_id, user_id, ["t1"], attempt_id)


def test_remediation_tags_filtered_to_attempts_actual_concept_tags():
    """Defense-in-depth mirroring the grades[]/question_id guard: a
    hallucinated/paraphrased concept tag from the LLM's free-form verdict
    must never reach remediate_chapter_task.delay(...) — only tags that
    were actually among the attempt's questions' concept_tag values."""
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-remediate-filter", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "wrong"})
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        user_id = attempt.user_id
        chapter_id = _chapter_id_for_assignment(db, assignment_id)

    fake_response = GradingResponse(
        grades=[], remediation_concept_tags=["t1", "invented-hallucinated-tag"],
        verdict_reasoning="Missed t1.",
    )
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response), \
         patch("app.agents.evaluation.grade.remediate_chapter_task") as mock_remediate_task:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    mock_remediate_task.delay.assert_called_once_with(chapter_id, user_id, ["t1"], attempt_id)


def test_remediation_not_dispatched_when_remediation_tags_empty():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-remediate-b", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a"})

    fake_response = GradingResponse(grades=[], remediation_concept_tags=[], verdict_reasoning="All good.")
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response), \
         patch("app.agents.evaluation.grade.remediate_chapter_task") as mock_remediate_task:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    mock_remediate_task.delay.assert_not_called()


def test_remediation_never_dispatched_for_module_level_attempt():
    with SessionLocal() as db:
        course = Course(topic_slug="grade-remediate-c", topic_raw="grade-remediate-c",
                         topic_embedding=[0.0] * 2048, created_at=_now())
        db.add(course)
        db.flush()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.flush()
        assignment = Assignment(level="module", module_id=module.id, scope="global",
                                 status="ready", created_at=_now(), updated_at=_now())
        db.add(assignment)
        db.flush()
        question = AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q",
                                       options=["a", "b"], correct_answer="a", explanation="e",
                                       concept_tag="t1", difficulty="easy")
        db.add(question)
        db.flush()
        user = User(email="grade-remediate-c@example.com", password_hash="x")
        db.add(user)
        db.flush()
        attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="grading",
                                     created_at=_now(), updated_at=_now())
        db.add(attempt)
        db.flush()
        db.add(AssignmentAnswer(attempt_id=attempt.id, question_id=question.id, concept_tag="t1", user_answer="wrong"))
        db.commit()
        attempt_id = attempt.id

    # Even though the LLM says t1 warrants remediation, module-level
    # assignments never dispatch it — Trigger A (chapter-level) only.
    fake_response = GradingResponse(grades=[], remediation_concept_tags=["t1"], verdict_reasoning="Missed t1.")
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response), \
         patch("app.agents.evaluation.grade.remediate_chapter_task") as mock_remediate_task:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    mock_remediate_task.delay.assert_not_called()
    with SessionLocal() as db:
        attempt_after = db.get(AssignmentAttempt, attempt_id)
        assert attempt_after is not None
        assert attempt_after.status == "graded"  # still grades normally, just no dispatch


def test_successful_grading_records_streak_activity():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-streak-a", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a"})
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        user_id = attempt.user_id

    fake_response = GradingResponse(grades=[], remediation_concept_tags=[], verdict_reasoning="ok")
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response):
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    with SessionLocal() as db:
        streak = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert streak.current_streak == 1
        assert streak.longest_streak == 1


def test_failed_grading_does_not_record_streak_activity():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-streak-b", [
            {"type": "free_text", "correct_answer": "X is Y", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "no idea"})
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        user_id = attempt.user_id

    with patch("app.agents.evaluation.grade.grade_assignment_answers", side_effect=RuntimeError("llm down")):
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    with SessionLocal() as db:
        streak = db.query(LearnerStreak).filter_by(user_id=user_id).one_or_none()
        assert streak is None


def test_second_grading_same_day_same_user_updates_existing_streak_row():
    """Grading two attempts for the SAME user on the SAME UTC day must go
    through record_activity's update branch (day_gap == 0) the second time,
    not attempt a second insert. Also exercises that update branch through
    the real grading trigger path."""
    with SessionLocal() as db:
        user = User(email="grade-streak-same-day@example.com", password_hash="x")
        db.add(user)
        db.flush()
        user_id = user.id

        assignment_id = _make_assignment_with_questions(db, "grade-streak-d", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
        ])
        questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment_id).order_by(AssignmentQuestion.order).all()

        attempt1 = AssignmentAttempt(assignment_id=assignment_id, user_id=user_id, status="grading",
                                      created_at=_now(), updated_at=_now())
        db.add(attempt1)
        db.flush()
        db.add(AssignmentAnswer(attempt_id=attempt1.id, question_id=questions[0].id,
                                 concept_tag=questions[0].concept_tag, user_answer="a"))

        attempt2 = AssignmentAttempt(assignment_id=assignment_id, user_id=user_id, status="grading",
                                      created_at=_now(), updated_at=_now())
        db.add(attempt2)
        db.flush()
        db.add(AssignmentAnswer(attempt_id=attempt2.id, question_id=questions[0].id,
                                 concept_tag=questions[0].concept_tag, user_answer="a"))
        db.commit()
        attempt1_id, attempt2_id = attempt1.id, attempt2.id

    fake_response = GradingResponse(grades=[], remediation_concept_tags=[], verdict_reasoning="ok")
    with patch("app.agents.evaluation.grade.grade_assignment_answers", return_value=fake_response):
        with SessionLocal() as db:
            grade_assignment_attempt(attempt1_id, db)
        with SessionLocal() as db:
            grade_assignment_attempt(attempt2_id, db)

    with SessionLocal() as db:
        attempt1_after = db.get(AssignmentAttempt, attempt1_id)
        attempt2_after = db.get(AssignmentAttempt, attempt2_id)
        assert attempt1_after is not None and attempt1_after.status == "graded"
        assert attempt2_after is not None and attempt2_after.status == "graded"
        streak = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert streak.current_streak == 1  # same UTC day both times — no streak change
        assert streak.longest_streak == 1


def test_idempotent_regrade_does_not_double_record_streak_activity():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-streak-c", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a"})
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        user_id = attempt.user_id  # read before commit expires/detaches `attempt`
        attempt.status = "graded"
        attempt.overall_score = 1.0
        db.commit()

    with SessionLocal() as db:
        grade_assignment_attempt(attempt_id, db)  # already graded — must no-op, including streak (early-return guard fires before any grader call, so no mock needed here)

    with SessionLocal() as db:
        streak = db.query(LearnerStreak).filter_by(user_id=user_id).one_or_none()
        assert streak is None
