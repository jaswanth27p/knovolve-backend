"""Tracking service: a learner's enrolled/tracked courses and dashboard."""
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.chapter_content import ChapterContent
from app.models.course import Course, Module, Chapter
from app.models.enrollment import UserCourse
from app.schemas.course import DashboardResponse, TrackedCourseResponse


def _serialize_tracked(db: Session, uc: UserCourse) -> TrackedCourseResponse:
    course = db.get(Course, uc.course_id)
    if course is None:
        raise HTTPException(status_code=500, detail="tracked course missing")
    modules = db.query(Module).filter_by(course_id=course.id).all()
    chapter_ids = [
        c.id for m in modules
        for c in db.query(Chapter).filter_by(module_id=m.id).all()
    ]
    ready_rows = 0
    if chapter_ids:
        ready_rows = db.query(ChapterContent).filter(
            ChapterContent.chapter_id.in_(chapter_ids),
            ChapterContent.scope == "global",
            ChapterContent.status == "ready",
        ).count()
    content_ready = len(chapter_ids) > 0 and ready_rows == len(chapter_ids)
    return TrackedCourseResponse(
        id=course.id, topic_slug=course.topic_slug, topic_raw=course.topic_raw,
        status=uc.status, progress=uc.progress, last_opened_at=uc.last_opened_at,
        module_count=len(modules), chapter_count=len(chapter_ids),
        content_ready=content_ready,
    )


def list_my_courses(db: Session, user_id: int) -> list[TrackedCourseResponse]:
    rows = db.query(UserCourse).filter_by(user_id=user_id).order_by(UserCourse.last_opened_at.desc()).all()
    return [_serialize_tracked(db, uc) for uc in rows]


def get_dashboard(db: Session, user_id: int) -> DashboardResponse:
    rows = db.query(UserCourse).filter_by(user_id=user_id).order_by(UserCourse.last_opened_at.desc()).all()
    in_progress = [_serialize_tracked(db, uc) for uc in rows if uc.status == "in_progress"]
    completed = [_serialize_tracked(db, uc) for uc in rows if uc.status == "completed"]
    return DashboardResponse(
        in_progress=in_progress, completed=completed,
        in_progress_count=len(in_progress), completed_count=len(completed),
        total_count=len(rows),
    )


def untrack_course(db: Session, user_id: int, course_id: int) -> None:
    row = db.query(UserCourse).filter_by(user_id=user_id, course_id=course_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="course not tracked")
    db.delete(row)
    db.commit()
