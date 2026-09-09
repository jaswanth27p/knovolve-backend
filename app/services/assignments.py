"""Assignment fetch-or-dispatch service: chapter/module assignment lookup,
generation dispatch, and course-scoped assignment resolution for attempts."""
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.assignment import Assignment, AssignmentQuestion, AssignmentUserTopup
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Course, Module
from app.schemas.assignment import AssignmentQuestionResponse, AssignmentResponse
from app.tasks.assignment_tasks import (
    generate_chapter_assignment_task,
    generate_module_assignment_task,
    generate_module_topup_task,
)


def _chapter_assignment(db: Session, chapter_content_id: int) -> Assignment | None:
    """The global chapter assignment for this content, if any. Filtered on
    level+scope the same way `_get_or_create_assignment` filters: the schema
    allows a per-user row on the same chapter_content_id, and an unfiltered
    single-row fetch would then be free to hand one learner another's
    assignment. Scope stays hardcoded to "global" — V1 only generates those."""
    return db.scalar(
        select(Assignment).where(
            Assignment.level == "chapter",
            Assignment.scope == "global",
            Assignment.chapter_content_id == chapter_content_id,
        )
    )


def _module_assignment(db: Session, module_id: int) -> Assignment | None:
    """The global module assignment for this module, if any. See
    `_chapter_assignment` for why level/scope are filtered."""
    return db.scalar(
        select(Assignment).where(
            Assignment.level == "module",
            Assignment.scope == "global",
            Assignment.module_id == module_id,
        )
    )


def _serialize_assignment(assignment: Assignment, db: Session, user_id: int | None = None) -> AssignmentResponse:
    """`user_id` is only ever passed by the module-assignment fetch path (to
    include that learner's own topup questions alongside the shared base
    set); the chapter-assignment path always calls this with the default
    `None`, since chapter-level AssignmentQuestion rows never get a user_id
    (Task 2 only ever writes user_id on module-level topup rows) and adding
    the OR-filter there would be a no-op at best."""
    if assignment.status != "ready":
        return AssignmentResponse(status=assignment.status, error=assignment.error)
    filters = [AssignmentQuestion.assignment_id == assignment.id]
    if user_id is not None:
        filters.append((AssignmentQuestion.user_id.is_(None)) | (AssignmentQuestion.user_id == user_id))
    questions = db.scalars(
        select(AssignmentQuestion)
        .where(*filters)
        .order_by(AssignmentQuestion.order)
    ).all()
    return AssignmentResponse(
        status="ready",
        questions=[
            AssignmentQuestionResponse(
                id=q.id, order=q.order, type=q.type, text=q.text, options=q.options,
                concept_tag=q.concept_tag, difficulty=q.difficulty,
            )
            for q in questions
        ],
    )


def _module_chapters_ready(module: Module, db: Session) -> bool:
    chapters = db.query(Chapter).filter_by(module_id=module.id).all()
    if not chapters:
        return False
    for chapter in chapters:
        content = db.scalar(
            select(ChapterContent).where(ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global")
        )
        if content is None or content.status != "ready":
            return False
    return True


def assignment_belongs_to_course(db: Session, assignment: Assignment, course_id: int) -> bool:
    """True if `assignment` (chapter- or module-level) is part of `course_id`.
    Used by both the submit and fetch attempt routes so an assignment id from
    a different course can never be submitted/fetched against this slug."""
    if assignment.level == "chapter":
        content = db.get(ChapterContent, assignment.chapter_content_id)
        if content is None:
            return False
        chapter = db.get(Chapter, content.chapter_id)
        if chapter is None:
            return False
        module = db.get(Module, chapter.module_id)
        return module is not None and module.course_id == course_id
    module = db.get(Module, assignment.module_id)
    return module is not None and module.course_id == course_id


def get_chapter_assignment(db: Session, chapter: Chapter) -> AssignmentResponse:
    """Fetch-or-dispatch the global assignment for `chapter`'s ready content.
    404s if the chapter's content isn't ready yet (nothing to base an
    assignment on); otherwise returns the existing assignment, dispatches
    generation and re-fetches if missing/failed, or reports "generating" if
    the dispatched task hasn't produced a row yet."""
    content = db.scalar(
        select(ChapterContent).where(ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global")
    )
    if content is None or content.status != "ready":
        raise HTTPException(status_code=404, detail="chapter content not ready")

    assignment = _chapter_assignment(db, content.id)
    if assignment is None or assignment.status == "failed":
        generate_chapter_assignment_task.delay(content.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = _chapter_assignment(db, content.id)
    if assignment is None:
        return AssignmentResponse(status="generating")
    return _serialize_assignment(assignment, db)


def create_module_assignment(db: Session, module: Module) -> AssignmentResponse:
    """Fetch-or-dispatch the global assignment for `module`. 409s if any of
    the module's chapters lack ready content."""
    if not _module_chapters_ready(module, db):
        raise HTTPException(status_code=409, detail="not all chapters in this module have ready content")

    assignment = _module_assignment(db, module.id)
    if assignment is None or assignment.status == "failed":
        generate_module_assignment_task.delay(module.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = _module_assignment(db, module.id)
    if assignment is None:
        return AssignmentResponse(status="generating")
    return _serialize_assignment(assignment, db)


def get_module_assignment(db: Session, module: Module, user_id: int) -> AssignmentResponse:
    """Fetch-or-dispatch the global assignment for `module`, same as
    `create_module_assignment` but only gates on chapter readiness (409) when
    a dispatch is actually needed (no existing/failed assignment found).

    Once the base assignment is "ready", also ensures `user_id`'s own topup
    exists (dispatching generation fire-and-forget on first fetch) and
    includes their topup questions once ready. This is deliberately
    invisible at the AssignmentResponse.status level in V1 — the base
    assignment being "ready" is what `status` has always meant here; a topup
    landing later doesn't change it, and there's no polling signal today for
    "new questions just landed" (that's a future UI, not built here)."""
    assignment = _module_assignment(db, module.id)
    if assignment is None or assignment.status == "failed":
        if not _module_chapters_ready(module, db):
            raise HTTPException(status_code=409, detail="not all chapters in this module have ready content")
        generate_module_assignment_task.delay(module.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = _module_assignment(db, module.id)
    if assignment is None:
        return AssignmentResponse(status="generating")

    if assignment.status == "ready":
        topup = db.scalar(
            select(AssignmentUserTopup).where(
                AssignmentUserTopup.assignment_id == assignment.id, AssignmentUserTopup.user_id == user_id,
            )
        )
        if topup is None:
            now = datetime.now(timezone.utc)
            db.add(AssignmentUserTopup(
                assignment_id=assignment.id, user_id=user_id, status="generating", created_at=now, updated_at=now,
            ))
            db.commit()
            generate_module_topup_task.delay(assignment.id, user_id)  # pyright: ignore[reportFunctionMemberAccess]

    return _serialize_assignment(assignment, db, user_id=user_id)


def get_assignment_for_course(db: Session, assignment_id: int, course: Course) -> Assignment:
    """Look up the global-scope assignment `assignment_id`, 404ing if it
    doesn't exist or doesn't belong to `course` (via `assignment_belongs_to_course`).
    Deliberately does NOT check `assignment.status` — this mirrors the current
    get_assignment_attempt lookup, which doesn't gate on status either; only
    submit_assignment_attempt adds its own extra "status == ready" check on
    top of this lookup. Callers that need that gate must add it themselves."""
    assignment = db.scalar(
        select(Assignment).where(Assignment.id == assignment_id, Assignment.scope == "global")
    )
    if assignment is None or not assignment_belongs_to_course(db, assignment, course.id):
        raise HTTPException(status_code=404, detail="assignment not found")
    return assignment
