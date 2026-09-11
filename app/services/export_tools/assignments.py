from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.chapter_content import ChapterContent
from app.services.assignments import (
    _chapter_assignment,
    _module_assignment,
    assignment_belongs_to_course,
)
from app.services.chapter_content import _visible_to_user
from app.services.export_tools import content as content_tools
from app.services.export_tools._authz import require_export_course
from app.services.export_tools._errors import ExportToolError


def _question_payload(q: AssignmentQuestion) -> dict:
    return {
        "id": q.id,
        "order": q.order,
        "type": q.type,
        "text": q.text,
        "options": q.options,
        "difficulty": q.difficulty,
        "concept_tag": q.concept_tag,
        "correct_answer": q.correct_answer,
        "explanation": q.explanation,
    }


def list_assignments(db: Session, user_id: int, course_slug: str) -> list[dict]:
    course = require_export_course(db, user_id, course_slug)
    items = []
    for chapter in content_tools.list_all_chapters(db, user_id, course_slug):
        versions = db.scalars(
            select(ChapterContent)
            .where(
                ChapterContent.chapter_id == chapter["chapter_id"],
                _visible_to_user(user_id),
            )
            .order_by(ChapterContent.version)
        ).all()
        for content in versions:
            assignment = _chapter_assignment(db, content)
            items.append({
                "assignment_id": assignment.id if assignment else None,
                "chapter_id": chapter["chapter_id"],
                "chapter_title": chapter["title"],
                "version": content.version,
                "level": "chapter",
                "scope": "remediation" if content.remediation_source_attempt_id is not None
                else chapter["module_scope"],
                "status": assignment.status if assignment else "missing",
            })
    for module in content_tools.list_modules(db, user_id, course_slug):
        if module["scope"] != "global":
            continue
        assignment = _module_assignment(db, module["id"])
        items.append({
            "assignment_id": assignment.id if assignment else None,
            "module_id": module["id"],
            "module_title": module["title"],
            "version": None,
            "level": "module",
            "scope": "global",
            "status": assignment.status if assignment else "missing",
        })
    return items


def get_assignment_questions(db: Session, user_id: int, course_slug: str, assignment_id: int) -> dict:
    course = require_export_course(db, user_id, course_slug)
    assignment = db.get(Assignment, assignment_id)
    if assignment is None or not assignment_belongs_to_course(db, assignment, course.id):
        raise ExportToolError(f"No assignment {assignment_id} in this course.")
    if assignment.scope == "user" and assignment.user_id != user_id:
        raise ExportToolError(f"Assignment {assignment_id} is not available to you.")
    questions = db.scalars(
        select(AssignmentQuestion)
        .where(
            AssignmentQuestion.assignment_id == assignment.id,
            (AssignmentQuestion.user_id.is_(None)) | (AssignmentQuestion.user_id == user_id),
        )
        .order_by(AssignmentQuestion.order)
    ).all()
    return {
        "assignment_id": assignment.id,
        "level": assignment.level,
        "status": assignment.status,
        "questions": [_question_payload(q) for q in questions],
    }
