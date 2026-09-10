"""Course extension: the per-user 'Additional Chapters' bucket and its jobs."""
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.assignment import Assignment, AssignmentQuestion, AssignmentUserTopup
from app.models.attempt import AssignmentAnswer, AssignmentAttempt
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter, Course, Module
from app.models.course_extension import CourseExtensionJob

BUCKET_TITLE = "Additional Chapters"


def get_or_create_bucket(db: Session, user_id: int, course: Course) -> Module:
    bucket = db.scalar(
        select(Module).where(
            Module.course_id == course.id, Module.scope == "user", Module.user_id == user_id,
        )
    )
    if bucket is not None:
        return bucket
    max_global = db.scalar(
        select(Module.order).where(Module.course_id == course.id, Module.scope == "global")
        .order_by(Module.order.desc()).limit(1)
    ) or 0
    bucket = Module(
        course_id=course.id, title=BUCKET_TITLE,
        objective="Chapters you added to fill gaps in this course.",
        order=max_global + 1, scope="user", user_id=user_id,
    )
    db.add(bucket)
    db.flush()
    return bucket


def append_chapters(db: Session, user_id: int, course: Course,
                    drafts: list[dict]) -> list[dict]:
    """Append chapter skeletons under this user's bucket. Returns the result
    payload [{chapter_id, title, objective}]. Caller commits."""
    bucket = get_or_create_bucket(db, user_id, course)
    next_order = (
        db.scalar(
            select(Chapter.order).where(Chapter.module_id == bucket.id)
            .order_by(Chapter.order.desc()).limit(1)
        ) or 0
    ) + 1
    added = []
    for i, draft in enumerate(drafts, start=next_order):
        chapter = Chapter(module_id=bucket.id, title=draft["title"],
                          objective=draft["objective"], order=i,
                          scope="user", user_id=user_id)
        db.add(chapter)
        db.flush()
        added.append({"chapter_id": chapter.id, "title": chapter.title,
                      "objective": chapter.objective})
    return added


def create_extension_job(db: Session, user_id: int, course: Course, message: str) -> CourseExtensionJob:
    active = db.scalar(
        select(CourseExtensionJob).where(
            CourseExtensionJob.course_id == course.id,
            CourseExtensionJob.user_id == user_id,
            CourseExtensionJob.status.in_(["pending", "running"]),
        )
    )
    if active is not None:
        raise HTTPException(status_code=409, detail="an extension is already running for this course")
    now = datetime.now(timezone.utc)
    job = CourseExtensionJob(course_id=course.id, user_id=user_id, request=message,
                             status="pending", created_at=now, updated_at=now)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_extension_job(db: Session, user_id: int, course: Course, job_id: int) -> CourseExtensionJob:
    job = db.scalar(
        select(CourseExtensionJob).where(
            CourseExtensionJob.id == job_id,
            CourseExtensionJob.course_id == course.id,
            CourseExtensionJob.user_id == user_id,
        )
    )
    if job is None:
        raise HTTPException(status_code=404, detail="extension job not found")
    return job


def _bucket_chapters(db: Session, user_id: int, course: Course) -> list[tuple[Module, list[Chapter]]]:
    bucket = db.scalar(
        select(Module).where(
            Module.course_id == course.id, Module.scope == "user", Module.user_id == user_id,
        )
    )
    if bucket is None:
        return []
    chapters = db.query(Chapter).filter_by(module_id=bucket.id).order_by(Chapter.order).all()
    return [(bucket, chapters)]


def list_extension_chapters(db: Session, user_id: int, course: Course) -> list[dict]:
    out = []
    for _bucket, chapters in _bucket_chapters(db, user_id, course):
        for chapter in chapters:
            content = db.scalar(
                select(ChapterContent).where(
                    ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "user",
                    ChapterContent.user_id == user_id,
                    ChapterContent.remediation_source_attempt_id.is_(None),
                )
            )
            out.append({
                "id": chapter.id, "title": chapter.title, "objective": chapter.objective,
                "order": chapter.order,
                "content_ready": content is not None and content.status == "ready",
            })
    return out


def _delete_cascade(db: Session, chapter: Chapter) -> None:
    content_ids = list(db.scalars(
        select(ChapterContent.id).where(ChapterContent.chapter_id == chapter.id)
    ).all())
    assignment_ids = list(db.scalars(
        select(Assignment.id).where(
            Assignment.chapter_content_id.in_(content_ids) if content_ids else Assignment.id.is_(None)
        )
    ).all())
    if assignment_ids:
        attempt_ids = list(db.scalars(
            select(AssignmentAttempt.id).where(AssignmentAttempt.assignment_id.in_(assignment_ids))
        ).all())
        if attempt_ids:
            db.execute(delete(AssignmentAnswer).where(AssignmentAnswer.attempt_id.in_(attempt_ids)))
            db.execute(delete(AssignmentAttempt).where(AssignmentAttempt.id.in_(attempt_ids)))
        db.execute(delete(AssignmentQuestion).where(AssignmentQuestion.assignment_id.in_(assignment_ids)))
        db.execute(delete(AssignmentUserTopup).where(AssignmentUserTopup.assignment_id.in_(assignment_ids)))
        db.execute(delete(Assignment).where(Assignment.id.in_(assignment_ids)))
    if content_ids:
        db.execute(delete(ChapterContentSection).where(
            ChapterContentSection.chapter_content_id.in_(content_ids)))
        db.execute(delete(ChapterContent).where(ChapterContent.id.in_(content_ids)))
    db.execute(delete(Chapter).where(Chapter.id == chapter.id))


def delete_extension_chapter(db: Session, user_id: int, course: Course, chapter_id: int) -> None:
    chapter = db.get(Chapter, chapter_id)
    if chapter is None or chapter.scope != "user" or chapter.user_id != user_id:
        raise HTTPException(status_code=404, detail="extension chapter not found")
    module = db.get(Module, chapter.module_id)
    if module is None or module.course_id != course.id or module.user_id != user_id:
        raise HTTPException(status_code=404, detail="extension chapter not found")
    _delete_cascade(db, chapter)
    db.commit()
