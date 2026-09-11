"""Course-level progress/status, recomputed after every graded attempt.
"Passed" is a coarser pass/fail gate than mastery's weak/developing/strong
bands (app.services.mastery) — different question (is this CHAPTER done)
from mastery's (is this CONCEPT understood). See design doc §6."""
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Module
from app.models.enrollment import UserCourse

PASS_THRESHOLD = 0.7


def resolve_course_id(db: Session, assignment: Assignment) -> int | None:
    """Resolves the course_id an assignment belongs to, via explicit FK
    lookups — there are no ORM relationship() declarations on Course/Module/
    Chapter/Assignment to traverse."""
    if assignment.level == "chapter":
        if assignment.chapter_content_id is None:
            return None
        content = db.get(ChapterContent, assignment.chapter_content_id)
        if content is None:
            return None
        chapter = db.get(Chapter, content.chapter_id)
        if chapter is None:
            return None
        module = db.get(Module, chapter.module_id)
        if module is None:
            return None
        return module.course_id
    if assignment.level == "module":
        if assignment.module_id is None:
            return None
        module = db.get(Module, assignment.module_id)
        if module is None:
            return None
        return module.course_id
    return None


def _latest_attempt_score(db: Session, assignment_id: int, user_id: int) -> float | None:
    attempt = db.scalar(
        select(AssignmentAttempt)
        .where(
            AssignmentAttempt.assignment_id == assignment_id,
            AssignmentAttempt.user_id == user_id,
            AssignmentAttempt.status == "graded",
        )
        .order_by(AssignmentAttempt.created_at.desc())
        .limit(1)
    )
    return attempt.overall_score if attempt is not None else None


def _resolve_relevant_content(db: Session, chapter_id: int, user_id: int) -> ChapterContent | None:
    """Which ChapterContent version currently applies to this user for this
    chapter: their own latest scope="user" remediation version if one
    exists, else the chapter's scope="global" version."""
    user_content = db.scalar(
        select(ChapterContent)
        .where(
            ChapterContent.chapter_id == chapter_id,
            ChapterContent.scope == "user",
            ChapterContent.user_id == user_id,
        )
        .order_by(ChapterContent.version.desc())
        .limit(1)
    )
    if user_content is not None:
        return user_content
    return db.scalar(
        select(ChapterContent).where(ChapterContent.chapter_id == chapter_id, ChapterContent.scope == "global")
    )


def _chapter_passed(db: Session, chapter_id: int, user_id: int) -> bool:
    content = _resolve_relevant_content(db, chapter_id, user_id)
    if content is None:
        return False
    assignment = db.scalar(
        select(Assignment).where(
            Assignment.level == "chapter",
            Assignment.scope == content.scope,
            Assignment.chapter_content_id == content.id,
            *([Assignment.user_id == user_id] if content.scope == "user" else []),
        )
    )
    if assignment is None:
        return False
    score = _latest_attempt_score(db, assignment.id, user_id)
    return score is not None and score >= PASS_THRESHOLD


def update_course_progress(db: Session, user_id: int, course_id: int) -> None:
    uc = db.scalar(select(UserCourse).where(UserCourse.user_id == user_id, UserCourse.course_id == course_id))
    if uc is None:
        return  # no enrollment row yet — nothing to update

    chapter_ids = [
        c.id for c in db.scalars(
            select(Chapter).join(Module, Chapter.module_id == Module.id).where(
                Module.course_id == course_id,
                or_(Module.scope == "global", and_(Module.scope == "user", Module.user_id == user_id)),
            )
        ).all()
    ]
    if not chapter_ids:
        return

    passed = sum(1 for cid in chapter_ids if _chapter_passed(db, cid, user_id))
    uc.progress = passed / len(chapter_ids)
    if uc.progress >= 1.0:
        uc.status = "completed"
    db.commit()
