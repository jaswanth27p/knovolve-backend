"""Export payloads, readiness gates, and user-scoped export-job helpers."""
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.documents.render import diagram_data_uri, markdown_to_html
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter, Course, Module
from app.models.export import ExportJob
from app.services.assignments import _chapter_assignment, _module_assignment
from app.storage import s3

EXPORT_KINDS = ("course", "assignments", "full_course", "full_assignments", "custom")
EXPORT_JOB_ERROR = "PDF export failed. Please try again."


def _now():
    return datetime.now(timezone.utc)


def global_modules(db: Session, course: Course) -> list[Module]:
    return db.query(Module).filter_by(course_id=course.id, scope="global").order_by(Module.order).all()


def user_bucket(db: Session, course: Course, user_id: int) -> Module | None:
    return db.scalar(
        select(Module).where(
            Module.course_id == course.id,
            Module.scope == "user",
            Module.user_id == user_id,
        )
    )


def chapters_in_scope(db: Session, course: Course, user_id: int, full: bool) -> list[Chapter]:
    module_ids = [m.id for m in global_modules(db, course)]
    if full:
        bucket = user_bucket(db, course, user_id)
        if bucket is not None:
            module_ids.append(bucket.id)
    return (
        db.query(Chapter)
        .filter(Chapter.module_id.in_(module_ids))
        .order_by(Chapter.module_id, Chapter.order)
        .all()
    )


def chapter_base_content(db: Session, chapter: Chapter, user_id: int) -> ChapterContent | None:
    if chapter.scope == "global":
        return db.scalar(
            select(ChapterContent).where(
                ChapterContent.chapter_id == chapter.id,
                ChapterContent.scope == "global",
            )
        )
    return db.scalar(
        select(ChapterContent).where(
            ChapterContent.chapter_id == chapter.id,
            ChapterContent.scope == "user",
            ChapterContent.user_id == user_id,
            ChapterContent.remediation_source_attempt_id.is_(None),
        )
    )


def chapter_remediation_versions(db: Session, chapter: Chapter, user_id: int) -> list[ChapterContent]:
    return list(db.scalars(
        select(ChapterContent)
        .where(
            ChapterContent.chapter_id == chapter.id,
            ChapterContent.scope == "user",
            ChapterContent.user_id == user_id,
            ChapterContent.remediation_source_attempt_id.isnot(None),
        )
        .order_by(ChapterContent.version)
    ).all())


def visible_versions(db: Session, chapter: Chapter, user_id: int, include_remediation: bool) -> list[ChapterContent]:
    versions = []
    base = chapter_base_content(db, chapter, user_id)
    if base is not None:
        versions.append(base)
    if include_remediation:
        versions.extend(chapter_remediation_versions(db, chapter, user_id))
    return versions


def _version_label(content: ChapterContent) -> str:
    if content.remediation_source_attempt_id is None:
        return f"Version {content.version}"
    return f"Personalized version {content.version}"


def _missing_global_content(db: Session, course: Course) -> list[int]:
    missing = []
    for module in global_modules(db, course):
        for chapter in db.query(Chapter).filter_by(module_id=module.id).order_by(Chapter.order).all():
            base = chapter_base_content(db, chapter, 0)
            if base is None or base.status != "ready":
                missing.append(chapter.id)
    return missing


def _incomplete_bucket_content(db: Session, course: Course, user_id: int) -> list[int]:
    bucket = user_bucket(db, course, user_id)
    if bucket is None:
        return []
    incomplete = []
    for chapter in db.query(Chapter).filter_by(module_id=bucket.id).order_by(Chapter.order).all():
        base = chapter_base_content(db, chapter, user_id)
        if base is None or base.status != "ready":
            incomplete.append(chapter.id)
    return incomplete


def _incomplete_remediation(db: Session, course: Course, user_id: int) -> list[int]:
    incomplete = []
    for chapter in chapters_in_scope(db, course, user_id, full=True):
        for content in chapter_remediation_versions(db, chapter, user_id):
            if content.status != "ready":
                incomplete.append(content.id)
    return incomplete


def _assignment_missing(db: Session, content: ChapterContent) -> bool:
    assignment = _chapter_assignment(db, content)
    return assignment is None or assignment.status != "ready"


def _missing_global_assignments(db: Session, course: Course) -> list[str]:
    missing = []
    for module in global_modules(db, course):
        chapters = db.query(Chapter).filter_by(module_id=module.id).order_by(Chapter.order).all()
        for chapter in chapters:
            base = chapter_base_content(db, chapter, 0)
            if base is None or _assignment_missing(db, base):
                missing.append(f"chapter:{chapter.id}")
        assignment = _module_assignment(db, module.id)
        if assignment is None or assignment.status != "ready":
            missing.append(f"module:{module.id}")
    return missing


def _missing_full_assignments(db: Session, course: Course, user_id: int) -> list[str]:
    missing = _missing_global_assignments(db, course)
    for chapter in chapters_in_scope(db, course, user_id, full=True):
        if chapter.scope == "global":
            contents = chapter_remediation_versions(db, chapter, user_id)
        else:
            base = chapter_base_content(db, chapter, user_id)
            contents = ([base] if base is not None else []) + chapter_remediation_versions(db, chapter, user_id)
        for content in contents:
            if content is not None and _assignment_missing(db, content):
                missing.append(f"content:{content.id}")
    return missing


def require_export_ready(db: Session, course: Course, user_id: int, kind: str) -> None:
    if kind not in EXPORT_KINDS:
        raise HTTPException(status_code=400, detail="unsupported export kind")
    if _missing_global_content(db, course):
        raise HTTPException(status_code=409, detail={
            "code": "content_not_ready",
            "message": "This course’s chapters are not fully generated yet.",
        })
    if kind == "course":
        return
    if kind == "assignments":
        if _missing_global_assignments(db, course):
            raise HTTPException(status_code=409, detail={
                "code": "assignments_not_ready",
                "message": "This course’s chapter and module assignments are not fully generated yet.",
            })
        return
    if _incomplete_bucket_content(db, course, user_id):
        raise HTTPException(status_code=409, detail={
            "code": "content_not_ready",
            "message": "Your additional chapters are not fully generated yet.",
        })
    if _incomplete_remediation(db, course, user_id):
        raise HTTPException(status_code=409, detail={
            "code": "versions_not_ready",
            "message": "One or more personalized versions are not fully generated yet.",
        })
    if kind in ("full_course", "custom"):
        return
    if _missing_full_assignments(db, course, user_id):
        raise HTTPException(status_code=409, detail={
            "code": "assignments_not_ready",
            "message": "One or more global or personalized assignments are not fully generated yet.",
        })


def readiness_summary(db: Session, course: Course, user_id: int) -> dict:
    global_content_ready = not _missing_global_content(db, course)
    bucket_content_ready = not _incomplete_bucket_content(db, course, user_id)
    versions_ready = not _incomplete_remediation(db, course, user_id)
    assignments_ready = not _missing_full_assignments(db, course, user_id)
    if not global_content_ready or not bucket_content_ready:
        status = "missing_content"
    elif not versions_ready:
        status = "missing_versions"
    elif not assignments_ready:
        status = "missing_assignments"
    else:
        status = "complete"
    return {
        "status": status,
        "global_content_ready": global_content_ready,
        "additional_content_ready": bucket_content_ready,
        "versions_ready": versions_ready,
        "assignments_ready": assignments_ready,
    }


def _version_sections(db: Session, content: ChapterContent) -> list[dict]:
    sections = db.scalars(
        select(ChapterContentSection)
        .where(ChapterContentSection.chapter_content_id == content.id)
        .order_by(ChapterContentSection.order)
    ).all()
    rendered = []
    for section in sections:
        rendered.append({
            "heading": section.heading,
            "kind": section.kind,
            "body_html": markdown_to_html(section.body_markdown),
            "examples": section.examples or [],
            "diagram_data_uri": diagram_data_uri(section.diagram_image_url),
        })
    return rendered


def _content_assignment_with_key(db: Session, content: ChapterContent, label: str) -> dict:
    assignment = _chapter_assignment(db, content)
    if assignment is None or assignment.status != "ready":
        raise HTTPException(status_code=409, detail={
            "code": "assignments_not_ready",
            "message": "One or more assignments are not fully generated yet.",
        })
    questions = db.scalars(
        select(AssignmentQuestion)
        .where(AssignmentQuestion.assignment_id == assignment.id)
        .order_by(AssignmentQuestion.order)
    ).all()
    return {
        "label": label,
        "level": "chapter",
        "questions": [
            {
                "order": q.order,
                "type": q.type,
                "text": q.text,
                "options": q.options,
                "difficulty": q.difficulty,
                "concept_tag": q.concept_tag,
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
            }
            for q in questions
        ],
    }


def _module_assignment_with_key(db: Session, module: Module) -> dict | None:
    assignment = _module_assignment(db, module.id)
    if assignment is None or assignment.status != "ready":
        raise HTTPException(status_code=409, detail={
            "code": "assignments_not_ready",
            "message": "One or more module assignments are not fully generated yet.",
        })
    questions = db.scalars(
        select(AssignmentQuestion)
        .where(
            AssignmentQuestion.assignment_id == assignment.id,
            AssignmentQuestion.user_id.is_(None),
        )
        .order_by(AssignmentQuestion.order)
    ).all()
    return {
        "label": f"{module.title} — Module assignment",
        "questions": [
            {
                "order": q.order,
                "type": q.type,
                "text": q.text,
                "options": q.options,
                "difficulty": q.difficulty,
                "concept_tag": q.concept_tag,
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
            }
            for q in questions
        ],
    }


def gather_course_payload(db: Session, course: Course, user_id: int, full: bool) -> dict:
    require_export_ready(db, course, user_id, "full_course" if full else "course")
    modules = []
    for module in global_modules(db, course):
        chapters = db.query(Chapter).filter_by(module_id=module.id).order_by(Chapter.order).all()
        modules.append({
            "title": module.title,
            "objective": module.objective,
            "is_additional": False,
            "chapters": [
                {
                    "title": chapter.title,
                    "objective": chapter.objective,
                    "versions": [
                        {
                            "label": _version_label(content),
                            "is_remediation": content.remediation_source_attempt_id is not None,
                            "sections": _version_sections(db, content),
                        }
                        for content in visible_versions(db, chapter, user_id, include_remediation=full)
                    ],
                }
                for chapter in chapters
            ],
        })
    if full:
        bucket = user_bucket(db, course, user_id)
        if bucket is not None:
            chapters = db.query(Chapter).filter_by(module_id=bucket.id).order_by(Chapter.order).all()
            modules.append({
                "title": bucket.title,
                "objective": bucket.objective,
                "is_additional": True,
                "chapters": [
                    {
                        "title": chapter.title,
                        "objective": chapter.objective,
                        "versions": [
                            {
                                "label": _version_label(content),
                                "is_remediation": content.remediation_source_attempt_id is not None,
                                "sections": _version_sections(db, content),
                            }
                            for content in visible_versions(db, chapter, user_id, include_remediation=True)
                        ],
                    }
                    for chapter in chapters
                ],
            })
    return {
        "title": course.topic_raw,
        "subtitle": "Full course" if full else "Export course",
        "generated_at": _now().isoformat(),
        "modules": modules,
    }


def gather_assignments_payload(db: Session, course: Course, user_id: int, full: bool) -> dict:
    require_export_ready(db, course, user_id, "full_assignments" if full else "assignments")
    modules = []
    for module in global_modules(db, course):
        chapters = db.query(Chapter).filter_by(module_id=module.id).order_by(Chapter.order).all()
        chapter_payloads = []
        for chapter in chapters:
            contents = visible_versions(db, chapter, user_id, include_remediation=full)
            chapter_payloads.append({
                "title": chapter.title,
                "objective": chapter.objective,
                "assignments": [
                    _content_assignment_with_key(db, content, _version_label(content))
                    for content in contents
                ],
            })
        modules.append({
            "title": module.title,
            "is_additional": False,
            "chapters": chapter_payloads,
            "module_assignment": _module_assignment_with_key(db, module),
        })
    if full:
        bucket = user_bucket(db, course, user_id)
        if bucket is not None:
            chapters = db.query(Chapter).filter_by(module_id=bucket.id).order_by(Chapter.order).all()
            modules.append({
                "title": bucket.title,
                "is_additional": True,
                "chapters": [
                    {
                        "title": chapter.title,
                        "objective": chapter.objective,
                        "assignments": [
                            _content_assignment_with_key(db, content, _version_label(content))
                            for content in visible_versions(db, chapter, user_id, include_remediation=True)
                        ],
                    }
                    for chapter in chapters
                ],
                "module_assignment": None,
            })
    return {
        "title": course.topic_raw,
        "subtitle": "Full assignments" if full else "Export assignments",
        "generated_at": _now().isoformat(),
        "modules": modules,
    }


def create_export_job(
    db: Session, user_id: int, course: Course, kind: str, params: dict | None = None,
) -> ExportJob:
    require_export_ready(db, course, user_id, kind)
    now = _now()
    job = ExportJob(
        course_id=course.id,
        user_id=user_id,
        kind=kind,
        status="pending",
        params=params,
        created_at=now,
        updated_at=now,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_export_job(db: Session, user_id: int, course: Course, export_id: int) -> ExportJob:
    job = db.scalar(
        select(ExportJob).where(
            ExportJob.id == export_id,
            ExportJob.course_id == course.id,
            ExportJob.user_id == user_id,
        )
    )
    if job is None:
        raise HTTPException(status_code=404, detail="export not found")
    return job


def list_export_jobs(db: Session, user_id: int, course: Course) -> list[ExportJob]:
    return list(db.scalars(
        select(ExportJob)
        .where(ExportJob.course_id == course.id, ExportJob.user_id == user_id)
        .order_by(ExportJob.created_at.desc())
    ).all())


def presigned_export_url(
    db: Session, user_id: int, course: Course, export_id: int, expires: int = 300,
) -> str:
    job = get_export_job(db, user_id, course, export_id)
    if job.status != "succeeded" or not job.result_key:
        raise HTTPException(status_code=409, detail={
            "code": "export_not_ready",
            "message": "This PDF is not ready for download yet.",
        })
    return s3.presign_get_url(job.result_key, expires=expires)
