from datetime import datetime, timezone
import pytest
from sqlalchemy.exc import IntegrityError
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.user import User
from app.models.attempt import AssignmentAttempt, AssignmentAnswer


def _now():
    return datetime.now(timezone.utc)


def _make_ready_assignment(db, slug: str) -> tuple[int, int]:
    """Returns (assignment_id, question_id)."""
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
    question = AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q",
                                   options=["a", "b"], correct_answer="a", explanation="e",
                                   concept_tag="t", difficulty="easy")
    db.add(question)
    db.flush()
    return assignment.id, question.id


def test_multiple_attempts_allowed_on_same_assignment():
    with SessionLocal() as db:
        assignment_id, _ = _make_ready_assignment(db, "att-model-a")
        user = User(email="att-model-a@example.com", password_hash="x")
        db.add(user)
        db.flush()
        db.add(AssignmentAttempt(assignment_id=assignment_id, user_id=user.id, status="grading",
                                  created_at=_now(), updated_at=_now()))
        db.add(AssignmentAttempt(assignment_id=assignment_id, user_id=user.id, status="grading",
                                  created_at=_now(), updated_at=_now()))
        db.commit()  # must not raise — no uniqueness constraint on (assignment_id, user_id)
        count = db.query(AssignmentAttempt).filter_by(assignment_id=assignment_id).count()
        assert count == 2


def test_unique_answer_per_question_per_attempt():
    with SessionLocal() as db:
        assignment_id, question_id = _make_ready_assignment(db, "att-model-b")
        user = User(email="att-model-b@example.com", password_hash="x")
        db.add(user)
        db.flush()
        attempt = AssignmentAttempt(assignment_id=assignment_id, user_id=user.id, status="grading",
                                     created_at=_now(), updated_at=_now())
        db.add(attempt)
        db.commit()
        db.add(AssignmentAnswer(attempt_id=attempt.id, question_id=question_id, concept_tag="t",
                                 user_answer="a"))
        db.commit()
        db.add(AssignmentAnswer(attempt_id=attempt.id, question_id=question_id, concept_tag="t",
                                 user_answer="b"))
        with pytest.raises(IntegrityError):
            db.commit()


def test_answer_defaults_ungraded():
    with SessionLocal() as db:
        assignment_id, question_id = _make_ready_assignment(db, "att-model-c")
        user = User(email="att-model-c@example.com", password_hash="x")
        db.add(user)
        db.flush()
        attempt = AssignmentAttempt(assignment_id=assignment_id, user_id=user.id, status="grading",
                                     created_at=_now(), updated_at=_now())
        db.add(attempt)
        db.commit()
        answer = AssignmentAnswer(attempt_id=attempt.id, question_id=question_id, concept_tag="t",
                                   user_answer="a")
        db.add(answer)
        db.commit()
        db.refresh(answer)
        assert answer.is_correct is None
        assert answer.feedback is None
        assert answer.graded_at is None
        assert answer.misconception_tag is None


def test_attempt_defaults_grading_status_and_null_score():
    with SessionLocal() as db:
        assignment_id, _ = _make_ready_assignment(db, "att-model-d")
        user = User(email="att-model-d@example.com", password_hash="x")
        db.add(user)
        db.flush()
        attempt = AssignmentAttempt(assignment_id=assignment_id, user_id=user.id,
                                     created_at=_now(), updated_at=_now())
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
        assert attempt.status == "grading"
        assert attempt.overall_score is None
        assert attempt.verdict_reasoning is None
