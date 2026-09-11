"""Tracking service: a learner's enrolled/tracked courses and dashboard."""
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.attempt import AssignmentAttempt
from app.models.course import Course, Module, Chapter
from app.models.enrollment import UserCourse
from app.models.learner_streak import LearnerStreak
from app.schemas.course import (
    ActivityEvent,
    ActivityResponse,
    DashboardResponse,
    StreakResponse,
    TrackedCourseResponse,
)
from app.services import mastery
from app.services.progression import PASS_THRESHOLD


def _serialize_tracked(db: Session, uc: UserCourse) -> TrackedCourseResponse:
    course = db.get(Course, uc.course_id)
    if course is None:
        raise HTTPException(status_code=500, detail="tracked course missing")
    from app.services.courses import visible_module_filter
    from app.services.progression import _resolve_relevant_content
    modules = [
        m for m in db.query(Module).filter(*visible_module_filter(course.id, uc.user_id)).all()
        if m.scope != "user" or db.query(Chapter).filter_by(module_id=m.id).count() > 0
    ]
    chapter_ids = [
        c.id for m in modules
        for c in db.query(Chapter).filter_by(module_id=m.id).all()
    ]
    ready_rows = 0
    for cid in chapter_ids:
        content = _resolve_relevant_content(db, cid, uc.user_id)
        if content is not None and content.status == "ready":
            ready_rows += 1
    content_ready = len(chapter_ids) > 0 and ready_rows == len(chapter_ids)
    statuses = mastery.get_concept_statuses(db, uc.user_id, course.id).values()
    return TrackedCourseResponse(
        id=course.id, topic_slug=course.topic_slug, topic_raw=course.topic_raw,
        status=uc.status, progress=uc.progress, last_opened_at=uc.last_opened_at,
        module_count=len(modules), chapter_count=len(chapter_ids),
        content_ready=content_ready,
        weak_concept_count=sum(1 for s in statuses if s == "weak"),
        strong_concept_count=sum(1 for s in statuses if s == "strong"),
    )


def list_my_courses(
    db: Session, user_id: int, *,
    search: str | None = None,
    status: str | None = None,
    sort: str = "date",
    order: str = "desc",
    page: int = 1,
    limit: int = 20,
) -> tuple[list[TrackedCourseResponse], int]:
    query = select(UserCourse).join(Course, Course.id == UserCourse.course_id).where(
        UserCourse.user_id == user_id
    )
    if status:
        query = query.where(UserCourse.status == status)
    if search:
        query = query.where(Course.topic_raw.ilike(f"%{search}%"))

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    if sort == "name":
        sort_column = Course.topic_raw
    elif sort == "progress":
        sort_column = UserCourse.progress
    else:
        sort_column = UserCourse.last_opened_at
    query = query.order_by(sort_column.asc() if order == "asc" else sort_column.desc())
    query = query.offset((page - 1) * limit).limit(limit)

    rows = db.scalars(query).all()
    return [_serialize_tracked(db, uc) for uc in rows], total


def get_dashboard(db: Session, user_id: int) -> DashboardResponse:
    rows = db.query(UserCourse).filter_by(user_id=user_id).order_by(UserCourse.last_opened_at.desc()).all()
    in_progress = [_serialize_tracked(db, uc) for uc in rows if uc.status == "in_progress"]
    completed = [_serialize_tracked(db, uc) for uc in rows if uc.status == "completed"]
    streak = db.query(LearnerStreak).filter_by(user_id=user_id).first()
    return DashboardResponse(
        in_progress=in_progress, completed=completed,
        in_progress_count=len(in_progress), completed_count=len(completed),
        total_count=len(rows),
        streak=StreakResponse(
            current=streak.current_streak if streak else 0,
            longest=streak.longest_streak if streak else 0,
        ),
    )


def untrack_course(db: Session, user_id: int, course_id: int) -> None:
    row = db.query(UserCourse).filter_by(user_id=user_id, course_id=course_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="course not tracked")
    db.delete(row)
    db.commit()


def get_activity(db: Session, user_id: int, days: int = 14) -> ActivityResponse:
    """Raw graded attempts for `days` local calendar days. Fetches `days + 1`
    UTC days so the browser's local-time bucketing never misses edge events at
    the UTC/local day boundary (a local day can start ~1 UTC day before/after
    its UTC equivalent)."""
    since = datetime.now(timezone.utc) - timedelta(days=days + 1)
    rows = db.scalars(
        select(AssignmentAttempt)
        .where(
            AssignmentAttempt.user_id == user_id,
            AssignmentAttempt.status == "graded",
            AssignmentAttempt.created_at >= since,
        )
        .order_by(AssignmentAttempt.created_at.asc())
    ).all()
    events = [
        ActivityEvent(
            at=attempt.created_at,
            score=attempt.overall_score if attempt.overall_score is not None else 0.0,
            passed=attempt.overall_score is not None and attempt.overall_score >= PASS_THRESHOLD,
        )
        for attempt in rows
    ]
    return ActivityResponse(days=days, events=events)
