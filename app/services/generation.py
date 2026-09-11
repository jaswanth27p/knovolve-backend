"""Full-course readiness computation and race-safe completion-run planning."""
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.course import Chapter, Course, Module
from app.models.export import CourseGenerationRun
from app.services import exports as export_service
from app.services.assignments import _module_assignment, _module_chapters_ready


def _now():
    return datetime.now(timezone.utc)


def readiness_summary(db: Session, course: Course, user_id: int) -> dict:
    return export_service.readiness_summary(db, course, user_id)


def get_latest_run(db: Session, user_id: int, course: Course) -> CourseGenerationRun | None:
    return db.scalar(
        select(CourseGenerationRun)
        .where(
            CourseGenerationRun.course_id == course.id,
            CourseGenerationRun.user_id == user_id,
        )
        .order_by(CourseGenerationRun.created_at.desc())
    )


def _assignment_incomplete(db: Session, content_id: int, scope: str, user_id: int | None) -> bool:
    from app.models.assignment import Assignment
    query = select(Assignment).where(
        Assignment.level == "chapter",
        Assignment.scope == scope,
        Assignment.chapter_content_id == content_id,
    )
    if scope == "user":
        query = query.where(Assignment.user_id == user_id)
    assignment = db.scalar(query)
    return assignment is None or assignment.status != "ready"


def _content_has_pending_diagrams(db: Session, content_id: int) -> bool:
    from app.models.chapter_content import ChapterContentSection
    return db.scalar(
        select(ChapterContentSection.id)
        .where(
            ChapterContentSection.chapter_content_id == content_id,
            ChapterContentSection.diagram_status == "pending",
        )
        .limit(1)
    ) is not None


def _chapter_inputs(db: Session, chapter: Chapter, user_id: int) -> list[dict]:
    units = []
    base = export_service.chapter_base_content(db, chapter, user_id)
    if base is None:
        if chapter.scope == "global" or chapter.user_id == user_id:
            units.append({
                "unit_id": f"content:{chapter.id}",
                "kind": "content",
                "chapter_id": chapter.id,
            })
        return units
    if base.status != "ready" and base.remediation_source_attempt_id is None:
        units.append({
            "unit_id": f"content:{chapter.id}",
            "kind": "content",
            "chapter_id": chapter.id,
        })
    elif base.status == "ready":
        # A ready row can still have `diagram_status="pending"` sections left by
        # the streaming generator; a content unit re-runs `ensure_chapter_content`,
        # which finalizes those diagrams and no-ops the already-persisted content.
        if _content_has_pending_diagrams(db, base.id):
            units.append({
                "unit_id": f"content:{chapter.id}",
                "kind": "content",
                "chapter_id": chapter.id,
            })
        if _assignment_incomplete(db, base.id, base.scope, user_id):
            units.append({
                "unit_id": f"chapter_assignment:{base.id}",
                "kind": "chapter_assignment",
                "chapter_id": chapter.id,
                "content_id": base.id,
            })
    for content in export_service.chapter_remediation_versions(db, chapter, user_id):
        if content.status != "ready":
            units.append({
                "unit_id": f"remediation_content:{content.id}",
                "kind": "remediation_content",
                "chapter_id": chapter.id,
                "content_id": content.id,
            })
        else:
            if _content_has_pending_diagrams(db, content.id):
                units.append({
                    "unit_id": f"remediation_content:{content.id}",
                    "kind": "remediation_content",
                    "chapter_id": chapter.id,
                    "content_id": content.id,
                })
            if _assignment_incomplete(db, content.id, "user", user_id):
                units.append({
                    "unit_id": f"remediation_assignment:{content.id}",
                    "kind": "remediation_assignment",
                    "chapter_id": chapter.id,
                    "content_id": content.id,
                })
    return units


def _module_inputs(db: Session, module: Module, user_id: int) -> list[dict]:
    units = []
    chapters = db.query(Chapter).filter_by(module_id=module.id).order_by(Chapter.order).all()
    for chapter in chapters:
        units.extend(_chapter_inputs(db, chapter, user_id))
    # Module assignments are global-only and are generated only from complete
    # chapter content, matching the existing module-assignment invariant.
    if module.scope == "global" and _module_chapters_ready(module, db):
        assignment = _module_assignment(db, module.id)
        if assignment is None or assignment.status != "ready":
            units.append({
                "unit_id": f"module_assignment:{module.id}",
                "kind": "module_assignment",
                "module_id": module.id,
            })
    return units


def plan_units(db: Session, course: Course, user_id: int) -> list[dict]:
    units = []
    for module in export_service.global_modules(db, course):
        units.extend(_module_inputs(db, module, user_id))
    bucket = export_service.user_bucket(db, course, user_id)
    if bucket is not None:
        for chapter in db.query(Chapter).filter_by(module_id=bucket.id).order_by(Chapter.order).all():
            units.extend(_chapter_inputs(db, chapter, user_id))
    return units


def queue_generation_run(db: Session, user_id: int, course: Course) -> tuple[CourseGenerationRun, str]:
    active = db.scalar(
        select(CourseGenerationRun).where(
            CourseGenerationRun.course_id == course.id,
            CourseGenerationRun.user_id == user_id,
            CourseGenerationRun.status.in_(["pending", "running"]),
        )
    )
    if active is not None:
        return active, "already_running"
    units = plan_units(db, course, user_id)
    now = _now()
    if not units:
        run = CourseGenerationRun(
            course_id=course.id,
            user_id=user_id,
            status="succeeded",
            total_units=0,
            completed_units=0,
            unit_states=[],
            created_at=now,
            updated_at=now,
            completed_at=now,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run, "already_complete"
    run = CourseGenerationRun(
        course_id=course.id,
        user_id=user_id,
        status="pending",
        total_units=len(units),
        completed_units=0,
        unit_states=[{**unit, "status": "pending"} for unit in units],
        created_at=now,
        updated_at=now,
    )
    db.add(run)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        active = db.scalar(
            select(CourseGenerationRun).where(
                CourseGenerationRun.course_id == course.id,
                CourseGenerationRun.user_id == user_id,
                CourseGenerationRun.status.in_(["pending", "running"]),
            )
        )
        if active is None:
            raise
        return active, "already_running"
    db.refresh(run)
    return run, "queued"


def advance_run(db: Session, run: CourseGenerationRun, unit_id: str, status: str) -> None:
    states = [dict(state) for state in (run.unit_states or [])]
    for state in states:
        if state.get("unit_id") == unit_id:
            state["status"] = status
    run.unit_states = states
    run.completed_units = sum(1 for state in states if state.get("status") in ("done", "failed"))
    run.updated_at = _now()
    db.commit()
