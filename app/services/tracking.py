"""Tracking service: a learner's enrolled/tracked courses and dashboard."""
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent
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

# Serializing the dashboard is O(queries) constant, but every serialized row
# still costs a fixed slice of work; cap how many enrollments a single
# dashboard call renders while the reported counts stay true totals.
DASHBOARD_COURSE_LIMIT = 100


def _serialize_tracked_bulk(
    db: Session, user_id: int, ucs: list[UserCourse]
) -> list[TrackedCourseResponse]:
    """Serialize many enrolled courses in a constant number of queries.

    The previous per-row serializer issued a handful of queries per course,
    per module, per chapter (plus a full mastery pass per course), which made
    ``list_my_courses``/``get_dashboard`` textbook N+1. This preloads every
    course, visible module, chapter, relevant content row, and concept status
    for the whole batch at once.
    """
    if not ucs:
        return []
    course_ids = list(dict.fromkeys(uc.course_id for uc in ucs))
    courses = {c.id: c for c in db.scalars(select(Course).where(Course.id.in_(course_ids))).all()}
    if len(courses) != len(course_ids):
        raise HTTPException(status_code=500, detail="tracked course missing")

    modules = db.scalars(
        select(Module).where(
            Module.course_id.in_(course_ids),
            or_(Module.scope == "global", and_(Module.scope == "user", Module.user_id == user_id)),
        ).order_by(Module.order)
    ).all()
    modules_by_course: dict[int, list[Module]] = {}
    for module in modules:
        modules_by_course.setdefault(module.course_id, []).append(module)

    chapter_rows = db.scalars(
        select(Chapter).where(Chapter.module_id.in_([m.id for m in modules])).order_by(Chapter.order)
    ).all() if modules else []
    chapters_by_module: dict[int, list[Chapter]] = {}
    for chapter in chapter_rows:
        chapters_by_module.setdefault(chapter.module_id, []).append(chapter)

    included_modules: dict[int, list[Module]] = {}
    chapter_ids_by_course: dict[int, list[int]] = {}
    all_chapter_ids: list[int] = []
    for course_id in course_ids:
        visible = [
            m for m in modules_by_course.get(course_id, [])
            if m.scope != "user" or chapters_by_module.get(m.id)
        ]
        included_modules[course_id] = visible
        ids = [c.id for m in visible for c in chapters_by_module.get(m.id, [])]
        chapter_ids_by_course[course_id] = ids
        all_chapter_ids.extend(ids)

    contents = db.scalars(
        select(ChapterContent).where(
            ChapterContent.chapter_id.in_(all_chapter_ids),
            or_(
                ChapterContent.scope == "global",
                and_(ChapterContent.scope == "user", ChapterContent.user_id == user_id),
            ),
        )
    ).all() if all_chapter_ids else []
    chosen = mastery.choose_relevant_contents(contents)
    status_counts = mastery.get_concept_status_counts_for_courses(db, user_id, course_ids)

    result: list[TrackedCourseResponse] = []
    for uc in ucs:
        course = courses[uc.course_id]
        ids = chapter_ids_by_course.get(uc.course_id, [])
        ready_rows = sum(
            1 for cid in ids
            if (content := chosen.get(cid)) is not None and content.status == "ready"
        )
        weak_count, strong_count = status_counts.get(uc.course_id, (0, 0))
        result.append(TrackedCourseResponse(
            id=course.id, topic_slug=course.topic_slug, topic_raw=course.topic_raw,
            status=uc.status, progress=uc.progress, last_opened_at=uc.last_opened_at,
            module_count=len(included_modules.get(uc.course_id, [])),
            chapter_count=len(ids),
            content_ready=len(ids) > 0 and ready_rows == len(ids),
            weak_concept_count=weak_count,
            strong_concept_count=strong_count,
        ))
    return result


def _serialize_tracked(db: Session, uc: UserCourse) -> TrackedCourseResponse:
    """Single-row serializer: routed through the bulk path so the two can never
    diverge."""
    return _serialize_tracked_bulk(db, uc.user_id, [uc])[0]


def get_tracked_course_by_slug(db: Session, user_id: int, slug: str) -> TrackedCourseResponse | None:
    """The single tracked-course row for (user, course slug), or None.

    Exists so the course-detail page can ask "is this one course tracked?"
    without fetching and paginating the learner's entire library.
    """
    uc = db.scalar(
        select(UserCourse)
        .join(Course, Course.id == UserCourse.course_id)
        .where(UserCourse.user_id == user_id, Course.topic_slug == slug)
    )
    if uc is None:
        return None
    return _serialize_tracked(db, uc)


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
    return _serialize_tracked_bulk(db, user_id, list(rows)), total


def get_dashboard(db: Session, user_id: int) -> DashboardResponse:
    # Cap the rows we serialize (recent first), but report true totals from an
    # aggregate so a learner with many enrollments never sees undercounted
    # progress just because the payload was bounded.
    rows = (
        db.query(UserCourse)
        .filter_by(user_id=user_id)
        .order_by(UserCourse.last_opened_at.desc())
        .limit(DASHBOARD_COURSE_LIMIT)
        .all()
    )
    in_progress = _serialize_tracked_bulk(
        db, user_id, [uc for uc in rows if uc.status == "in_progress"]
    )
    completed = _serialize_tracked_bulk(
        db, user_id, [uc for uc in rows if uc.status == "completed"]
    )
    status_counts: dict[str, int] = {}
    for status_value, status_count in db.execute(
        select(UserCourse.status, func.count())
        .where(UserCourse.user_id == user_id)
        .group_by(UserCourse.status)
    ).all():
        status_counts[status_value] = status_count
    streak = db.query(LearnerStreak).filter_by(user_id=user_id).first()
    return DashboardResponse(
        in_progress=in_progress, completed=completed,
        in_progress_count=status_counts.get("in_progress", 0),
        completed_count=status_counts.get("completed", 0),
        total_count=sum(status_counts.values()),
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
