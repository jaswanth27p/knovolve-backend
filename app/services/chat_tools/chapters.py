from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter, Module
from app.services import progression
from app.services.assignments import _chapter_assignment
from app.services.chat_tools._authz import require_started_course
from app.services.chat_tools._errors import ChatToolError


def _require_chapter_in_course(db: Session, course_id: int, chapter_id: int) -> Chapter:
    chapter = db.get(Chapter, chapter_id)
    if chapter is None:
        raise ChatToolError(f"No chapter {chapter_id}.")
    module = db.get(Module, chapter.module_id)
    if module is None or module.course_id != course_id:
        raise ChatToolError(f"Chapter {chapter_id} is not part of this course.")
    return chapter


def get_chapter_progress(db: Session, user_id: int, course_slug: str, chapter_id: int) -> dict:
    course = require_started_course(db, user_id, course_slug)
    chapter = _require_chapter_in_course(db, course.id, chapter_id)

    version_count = db.query(ChapterContent).filter(
        ChapterContent.chapter_id == chapter_id,
        (ChapterContent.scope == "global")
        | ((ChapterContent.scope == "user") & (ChapterContent.user_id == user_id)),
    ).count()

    attempt_count = 0
    latest_score = None
    content = progression._resolve_relevant_content(db, chapter_id, user_id)
    if content is not None:
        assignment = _chapter_assignment(db, content)
        if assignment is not None:
            attempts = db.scalars(
                select(AssignmentAttempt).where(
                    AssignmentAttempt.assignment_id == assignment.id,
                    AssignmentAttempt.user_id == user_id,
                    AssignmentAttempt.status == "graded",
                ).order_by(AssignmentAttempt.created_at.desc())
            ).all()
            attempt_count = len(attempts)
            latest_score = attempts[0].overall_score if attempts else None

    return {
        "chapter_id": chapter_id, "title": chapter.title,
        "completed": progression._chapter_passed(db, chapter_id, user_id),
        "content_version_count": version_count,
        "attempt_count": attempt_count,
        "latest_score": latest_score,
    }


def get_chapter_content(db: Session, user_id: int, course_slug: str, chapter_id: int) -> dict:
    course = require_started_course(db, user_id, course_slug)
    _require_chapter_in_course(db, course.id, chapter_id)

    content = progression._resolve_relevant_content(db, chapter_id, user_id)
    if content is None or content.status != "ready":
        return {"available": False, "reason": "Chapter content has not been generated yet."}

    sections = db.scalars(
        select(ChapterContentSection)
        .where(ChapterContentSection.chapter_content_id == content.id)
        .order_by(ChapterContentSection.order)
    ).all()
    return {
        "available": True, "version": content.version,
        "sections": [{"heading": s.heading, "body_markdown": s.body_markdown} for s in sections],
    }
