from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAnswer, AssignmentAttempt
from app.models.chapter_content import ChapterContent
from app.models.course import Course
from app.services import progression
from app.services.chapter_content import _visible_to_user
from app.services.chat_tools._authz import require_started, require_started_course
from app.services.chat_tools._errors import ChatToolError
from app.services.chat_tools.chapters import _require_chapter_in_course


def _course_for_assignment(db: Session, assignment: Assignment) -> Course:
    course_id = progression.resolve_course_id(db, assignment)
    course = db.get(Course, course_id) if course_id is not None else None
    if course is None:
        raise ChatToolError(f"Could not resolve the course for assignment {assignment.id}.")
    return course


def get_assignment(db: Session, user_id: int, assignment_id: int) -> dict:
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise ChatToolError(f"No assignment {assignment_id}.")
    course = _course_for_assignment(db, assignment)
    require_started(db, user_id, course)
    return {
        "id": assignment.id, "level": assignment.level, "status": assignment.status,
        "course_slug": course.topic_slug,
    }


def list_assignment_attempts(db: Session, user_id: int, assignment_id: int) -> list[dict]:
    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise ChatToolError(f"No assignment {assignment_id}.")
    course = _course_for_assignment(db, assignment)
    require_started(db, user_id, course)
    attempts = db.scalars(
        select(AssignmentAttempt)
        .where(AssignmentAttempt.assignment_id == assignment_id, AssignmentAttempt.user_id == user_id)
        .order_by(AssignmentAttempt.created_at.desc())
    ).all()
    return [
        {"id": a.id, "status": a.status, "overall_score": a.overall_score, "created_at": a.created_at.isoformat()}
        for a in attempts
    ]


def get_attempt_detail(db: Session, user_id: int, attempt_id: int) -> dict:
    attempt = db.get(AssignmentAttempt, attempt_id)
    if attempt is None or attempt.user_id != user_id:
        raise ChatToolError(f"No attempt {attempt_id} for this learner.")
    answers = db.scalars(
        select(AssignmentAnswer).where(AssignmentAnswer.attempt_id == attempt_id)
    ).all()
    question_ids = [a.question_id for a in answers]
    questions = {
        q.id: q for q in db.scalars(
            select(AssignmentQuestion).where(AssignmentQuestion.id.in_(question_ids))
        ).all()
    } if question_ids else {}
    return {
        "id": attempt.id, "status": attempt.status, "overall_score": attempt.overall_score,
        "answers": [
            {
                "question_text": questions[a.question_id].text if a.question_id in questions else None,
                "user_answer": a.user_answer,
                "correct_answer": questions[a.question_id].correct_answer if a.question_id in questions else None,
                "is_correct": a.is_correct,
                "misconception_tag": a.misconception_tag,
            }
            for a in answers
        ],
    }


def get_question(db: Session, user_id: int, question_id: int) -> dict:
    question = db.get(AssignmentQuestion, question_id)
    if question is None:
        raise ChatToolError(f"No question {question_id}.")
    assignment = db.get(Assignment, question.assignment_id)
    if assignment is None:
        raise ChatToolError(f"No question {question_id}.")
    course = _course_for_assignment(db, assignment)
    require_started(db, user_id, course)
    return {
        "id": question.id, "type": question.type, "text": question.text, "options": question.options,
        "correct_answer": question.correct_answer, "explanation": question.explanation,
        "concept_tag": question.concept_tag, "difficulty": question.difficulty,
    }


def _content_for_version(db: Session, chapter_id: int, user_id: int, version: int | None) -> ChapterContent:
    if version is None:
        content = progression._resolve_relevant_content(db, chapter_id, user_id)
        if content is None:
            raise ChatToolError("Chapter content has not been generated yet.")
        return content
    content = db.scalar(
        select(ChapterContent).where(
            ChapterContent.chapter_id == chapter_id,
            ChapterContent.version == version,
            _visible_to_user(user_id),
        )
    )
    if content is None:
        raise ChatToolError(f"No version {version} of this chapter's content for you.")
    return content


def get_chapter_assignment(
    db: Session, user_id: int, course_slug: str, chapter_id: int, version: int | None = None,
) -> dict:
    course = require_started_course(db, user_id, course_slug)
    _require_chapter_in_course(db, course.id, chapter_id)
    content = _content_for_version(db, chapter_id, user_id, version)

    assignment = db.scalar(
        select(Assignment).where(
            Assignment.level == "chapter",
            Assignment.scope == content.scope,
            Assignment.chapter_content_id == content.id,
            *([Assignment.user_id == content.user_id] if content.scope == "user" else []),
        )
    )
    result = {
        "id": assignment.id if assignment is not None else None,
        "level": "chapter", "scope": content.scope,
        "content_version": content.version,
        "status": assignment.status if assignment is not None else "generating",
        "error": assignment.error if assignment is not None else None,
    }
    if assignment is None or assignment.status != "ready":
        return result
    questions = db.scalars(
        select(AssignmentQuestion)
        .where(AssignmentQuestion.assignment_id == assignment.id)
        .order_by(AssignmentQuestion.order)
    ).all()
    result["questions"] = [
        {"id": q.id, "order": q.order, "type": q.type, "text": q.text, "options": q.options,
         "concept_tag": q.concept_tag, "difficulty": q.difficulty}
        for q in questions
    ]
    return result
