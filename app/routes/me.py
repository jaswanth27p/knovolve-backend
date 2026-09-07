from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import get_session
from app.auth.dependencies import get_current_user
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.models.enrollment import UserCourse
from app.models.user import User
from app.schemas.course import DashboardResponse, TrackedCourseResponse

router = APIRouter(prefix="/me", tags=["me"])


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


@router.get("/courses", response_model=list[TrackedCourseResponse])
def get_my_courses(db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    rows = db.query(UserCourse).filter_by(user_id=user.id).order_by(UserCourse.last_opened_at.desc()).all()
    return [_serialize_tracked(db, uc) for uc in rows]


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    rows = db.query(UserCourse).filter_by(user_id=user.id).order_by(UserCourse.last_opened_at.desc()).all()
    in_progress = [_serialize_tracked(db, uc) for uc in rows if uc.status == "in_progress"]
    completed = [_serialize_tracked(db, uc) for uc in rows if uc.status == "completed"]
    return DashboardResponse(
        in_progress=in_progress, completed=completed,
        in_progress_count=len(in_progress), completed_count=len(completed),
        total_count=len(rows),
    )


@router.delete("/courses/{course_id}", status_code=204)
def delete_my_course(course_id: int, db: Session = Depends(get_session),
                     user: User = Depends(get_current_user)):
    row = db.query(UserCourse).filter_by(user_id=user.id, course_id=course_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="course not tracked")
    db.delete(row)
    db.commit()