from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt, AssignmentAnswer
from app.models.user import User
from app.agents.evaluation.nodes.grade_free_text_answers import AnswerGrade
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


def test_grades_mcq_and_true_false_by_exact_match_no_llm_call():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-a", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
            {"type": "true_false", "correct_answer": "true", "concept_tag": "t2"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a", 1: "false"})

    with patch("app.agents.evaluation.grade.grade_free_text_answers") as mock_grade_free_text:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    mock_grade_free_text.assert_not_called()
    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        assert attempt.status == "graded"
        assert attempt.overall_score == 0.5
        answers = db.query(AssignmentAnswer).filter_by(attempt_id=attempt_id).order_by(AssignmentAnswer.question_id).all()
        assert answers[0].is_correct is True
        assert answers[1].is_correct is False
        assert answers[0].graded_at is not None


def test_exact_match_is_case_insensitive_and_trimmed():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-b", [
            {"type": "true_false", "correct_answer": "True", "concept_tag": "t1"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "  true  "})

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

    fake_grades = [
        AnswerGrade(question_id=q0_id, is_correct=True, feedback="Correct."),
        AnswerGrade(question_id=q1_id, is_correct=False, feedback="Incorrect."),
    ]
    with patch("app.agents.evaluation.grade.grade_free_text_answers", return_value=fake_grades) as mock_grade_free_text:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    mock_grade_free_text.assert_called_once()
    assert len(mock_grade_free_text.call_args.args[0]) == 2  # both items in one batched call
    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        assert attempt.overall_score == 0.5
        answers = {a.question_id: a for a in db.query(AssignmentAnswer).filter_by(attempt_id=attempt_id).all()}
        assert answers[q0_id].is_correct is True
        assert answers[q0_id].feedback == "Correct."
        assert answers[q1_id].is_correct is False


def test_mixed_mcq_and_free_text_in_one_attempt():
    with SessionLocal() as db:
        assignment_id = _make_assignment_with_questions(db, "grade-d", [
            {"type": "mcq", "correct_answer": "a", "concept_tag": "t1"},
            {"type": "free_text", "correct_answer": "X is Y", "concept_tag": "t2"},
        ])
        attempt_id = _make_attempt(db, assignment_id, {0: "a", 1: "X is Y"})
        questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment_id).order_by(AssignmentQuestion.order).all()
        free_text_question_id = questions[1].id

    fake_grades = [AnswerGrade(question_id=free_text_question_id, is_correct=True, feedback="Correct.")]
    with patch("app.agents.evaluation.grade.grade_free_text_answers", return_value=fake_grades) as mock_grade_free_text:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    # Only the free_text item is sent to the LLM — mcq graded without it.
    sent_items = mock_grade_free_text.call_args.args[0]
    assert len(sent_items) == 1
    assert sent_items[0].question_id == free_text_question_id
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

    with patch("app.agents.evaluation.grade.grade_free_text_answers", side_effect=RuntimeError("llm down")):
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)

    with SessionLocal() as db:
        attempt = db.get(AssignmentAttempt, attempt_id)
        assert attempt is not None
        assert attempt.status == "failed"
        assert attempt.error == "llm down"
        assert attempt.overall_score is None
        # The mcq answer was graded in-session before the free_text call failed —
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
    partial_grades = [AnswerGrade(question_id=q1_id, is_correct=True, feedback="Correct.")]
    with patch("app.agents.evaluation.grade.grade_free_text_answers", return_value=partial_grades):
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

    with patch("app.agents.evaluation.grade.grade_free_text_answers") as mock_grade_free_text:
        with SessionLocal() as db:
            grade_assignment_attempt(attempt_id, db)  # already graded — must no-op

    mock_grade_free_text.assert_not_called()


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
