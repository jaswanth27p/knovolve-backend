from datetime import datetime, timezone

import pytest

from app.db import SessionLocal
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAnswer, AssignmentAttempt
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Course, Module
from app.models.enrollment import UserCourse
from app.models.user import User
from app.services.chat_tools import assignments as assignment_tools
from app.services.chat_tools._errors import ChatToolError


def _now():
    return datetime.now(timezone.utc)


def _make_started_assignment(db, slug: str, user_id: int):
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
    db.add(course); db.flush()
    module = Module(course_id=course.id, title="M", objective="o", order=1)
    db.add(module); db.flush()
    chapter = Chapter(module_id=module.id, title="C1", objective="o", order=1)
    db.add(chapter); db.flush()
    content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                              outline=[], created_at=_now(), updated_at=_now())
    db.add(content); db.flush()
    assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                             status="ready", created_at=_now(), updated_at=_now())
    db.add(assignment); db.flush()
    db.add(UserCourse(user_id=user_id, course_id=course.id, enrolled_at=_now(), last_opened_at=_now()))
    db.flush()
    return course, chapter, content, assignment


def test_get_assignment_requires_started_course():
    with SessionLocal() as db:
        user = User(email="at-a@example.com", password_hash="x")
        db.add(user); db.flush()
        _, _, _, assignment = _make_started_assignment(db, "at-course-a", user.id)
        db.commit()
        assignment_id = assignment.id

    with SessionLocal() as db:
        with pytest.raises(ChatToolError, match="haven't started"):
            assignment_tools.get_assignment(db, user_id=999, assignment_id=assignment_id)


def test_get_attempt_detail_returns_per_question_breakdown():
    with SessionLocal() as db:
        user = User(email="at-b@example.com", password_hash="x")
        db.add(user); db.flush()
        _, _, _, assignment = _make_started_assignment(db, "at-course-b", user.id)
        question = AssignmentQuestion(
            assignment_id=assignment.id, order=1, type="mcq", text="2+2?",
            options=["3", "4"], correct_answer="4", explanation="basic math",
            concept_tag="arithmetic", difficulty="easy",
        )
        db.add(question); db.flush()
        attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                     overall_score=0.0, created_at=_now(), updated_at=_now())
        db.add(attempt); db.flush()
        db.add(AssignmentAnswer(attempt_id=attempt.id, question_id=question.id, concept_tag="arithmetic",
                                 user_answer="3", is_correct=False, misconception_tag="off-by-one"))
        db.commit()
        user_id, attempt_id = user.id, attempt.id

    with SessionLocal() as db:
        result = assignment_tools.get_attempt_detail(db, user_id, attempt_id)
    assert result["answers"][0]["is_correct"] is False
    assert result["answers"][0]["misconception_tag"] == "off-by-one"
    assert result["answers"][0]["correct_answer"] == "4"


def test_get_attempt_detail_rejects_other_users_attempt():
    with SessionLocal() as db:
        user = User(email="at-c@example.com", password_hash="x")
        db.add(user); db.flush()
        _, _, _, assignment = _make_started_assignment(db, "at-course-c", user.id)
        attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                     overall_score=1.0, created_at=_now(), updated_at=_now())
        db.add(attempt); db.commit()
        attempt_id = attempt.id

    with SessionLocal() as db:
        with pytest.raises(ChatToolError):
            assignment_tools.get_attempt_detail(db, user_id=999, attempt_id=attempt_id)
