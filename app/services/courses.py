"""Course catalog service: course generation jobs, dedup, list/fetch courses."""
import logging
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import and_, func, inspect as sa_inspect, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.course_creation.checkpoints import purge_checkpoints
from app.agents.course_creation.nodes.normalize_topic import (
    _canonicalize,
    _slugify,
    find_existing,
)
from app.config import settings
from app.llm.factory import embed
from app.models.course import Course, CourseJob, Module, Chapter
from app.models.enrollment import UserCourse
from app.schemas.course import CourseJobResponse, CourseCandidate, MyCourseJobResponse, PublicCourseResponse
from app.services import course_preview
from app.services.enrollment import touch_enrollment
from app.tasks.course_creation_task import run_course_creation_job

logger = logging.getLogger(__name__)


def _db_course(db: Session, course: Course) -> Course | None:
    """Re-fetch a Course row inside the current session by its primary key.

    find_existing can hand back a Course instance detached from this session
    (it completed its own transaction). Reading attributes on a detached,
    expired instance raises DetachedInstanceError, so we re-select by identity
    instead of touching it.
    """
    identity = sa_inspect(course).identity
    if not identity:
        return None
    return db.get(Course, identity[0])


def visible_module_filter(course_id: int, user_id: int | None) -> list:
    """SQLAlchemy filter args selecting a course's modules visible to `user_id`:
    every global module, plus the caller's own user-scoped bucket."""
    if user_id is None:
        return [Module.course_id == course_id, Module.scope == "global"]
    return [
        Module.course_id == course_id,
        or_(Module.scope == "global", and_(Module.scope == "user", Module.user_id == user_id)),
    ]


def serialize_course(db: Session, course: Course, user_id: int | None = None) -> dict:
    modules = db.query(Module).filter(*visible_module_filter(course.id, user_id)) \
        .order_by(Module.order).all()
    result = []
    for m in modules:
        chapters = db.query(Chapter).filter_by(module_id=m.id).order_by(Chapter.order).all()
        if m.scope == "user" and not chapters:
            continue  # never surface an empty bucket
        result.append({
            "id": m.id, "title": m.title, "objective": m.objective,
            "is_additional": m.scope == "user",
            "chapters": [{"id": c.id, "title": c.title, "objective": c.objective} for c in chapters],
        })
    return {
        "id": course.id, "topic_slug": course.topic_slug, "topic_raw": course.topic_raw,
        "modules": result,
    }


def _canonical_topic(raw: str) -> str:
    """Canonicalize the topic; an LLM flap must not 500 the request, so fall
    back to the raw string (degraded dedup, still functional)."""
    try:
        return _canonicalize(raw)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to raw
        logger.warning("topic canonicalization failed; falling back to raw: %s", exc)
        return raw


def _existing_response(existing: Course | CourseJob, db: Session, user_id: int | None = None) -> CourseJobResponse:
    """Dedup hit -> response object. status="exists" (course found) or
    status="pending" (in-flight duplicate job); the route maps "exists" to HTTP
    200 and leaves the decorator's 202 default for "pending"."""
    if isinstance(existing, Course):
        course = _db_course(db, existing)
        if course is None:
            raise HTTPException(status_code=500, detail="existing course not found")
        return CourseJobResponse(
            status="exists", job_id=course.id, course=serialize_course(db, course, user_id)
        )
    return CourseJobResponse(status="pending", job_id=existing.id)


def get_course_by_slug(db: Session, slug: str) -> Course:
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    return course


def get_module(db: Session, course: Course, module_id: int, user_id: int | None = None) -> Module:
    module = db.query(Module).filter(
        *visible_module_filter(course.id, user_id), Module.id == module_id,
    ).first()
    if not module:
        raise HTTPException(status_code=404, detail="module not found")
    return module


def get_next_chapter_id(db: Session, chapter_id: int, user_id: int | None = None) -> int | None:
    """Chapter immediately after `chapter_id` in course learning order: the
    next chapter in the same module by `Chapter.order`, else the first
    chapter of the next module by `Module.order`. None if `chapter_id` is
    the course's last chapter (or doesn't resolve)."""
    chapter = db.get(Chapter, chapter_id)
    if chapter is None:
        return None
    module = db.get(Module, chapter.module_id)
    if module is None:
        return None
    if module.scope == "user" and module.user_id != user_id:
        return None
    next_in_module = db.scalar(
        select(Chapter)
        .where(Chapter.module_id == module.id, Chapter.order > chapter.order)
        .order_by(Chapter.order)
        .limit(1)
    )
    if next_in_module is not None:
        return next_in_module.id
    next_module = db.scalar(
        select(Module)
        .where(*visible_module_filter(module.course_id, user_id), Module.order > module.order)
        .order_by(Module.order)
        .limit(1)
    )
    if next_module is None:
        return None
    first_chapter = db.scalar(
        select(Chapter).where(Chapter.module_id == next_module.id).order_by(Chapter.order).limit(1)
    )
    return first_chapter.id if first_chapter is not None else None


def create_course_job(db: Session, user_id: int, topic_raw: str,
                      force: bool = False, search_token: str | None = None) -> CourseJobResponse:
    # One in-flight generation per user at a time — the frontend already
    # blocks the "create" UI while a job is running, so a second job for
    # this user here would only be reachable by racing/replaying the
    # request, not normal use.
    ongoing = db.scalar(
        select(CourseJob).where(
            CourseJob.created_by_user_id == user_id,
            CourseJob.status.in_(["pending", "running"]),
        )
    )
    if ongoing is not None:
        return CourseJobResponse(status="pending", job_id=ongoing.id)

    if force:
        return _execute_forced(db, user_id, topic_raw, search_token)

    # Canonicalize + embed ONCE, then show similar candidates instead of
    # silently auto-merging: the user decides whether to navigate to an
    # existing course or force-generate a new one.
    canonical = _canonical_topic(topic_raw)
    embedding = embed(canonical)
    token = course_preview.cache_key(topic_raw, user_id)
    course_preview.store_preview(token, canonical, embedding, topic_raw)

    matches = course_preview.ranked_candidates(
        db, embedding, limit=3, threshold=settings.topic_candidate_threshold
    )
    if matches:
        return CourseJobResponse(
            status="similar", search_token=token,
            candidates=[CourseCandidate(**m) for m in matches],
        )

    return _create_and_dispatch(db, user_id, topic_raw, canonical, embedding)


def _execute_forced(db: Session, user_id: int, topic_raw: str,
                    search_token: str | None) -> CourseJobResponse:
    """Force path: reuse the preview's canonicalization when the token
    resolves (deterministic, zero extra LLM/embedding calls), else recompute.
    Exact identity still wins over force — an existing course/job with the
    exact canonical slug is attached to, only similarity matching is bypassed."""
    cached = course_preview.load_preview(search_token) if search_token else None
    if cached:
        canonical = cached["canonical"]
        embedding = cached["embedding"]
        topic_raw = cached["topic_raw"]
    else:
        canonical = _canonical_topic(topic_raw)
        embedding = embed(canonical)

    slug = _slugify(canonical)
    existing_course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if existing_course is not None:
        course = _db_course(db, existing_course)
        if course is None:
            raise HTTPException(status_code=500, detail="existing course not found")
        touch_enrollment(db, user_id, course)
        return _existing_response(course, db, user_id)
    dup_job = db.scalar(
        select(CourseJob).where(
            CourseJob.topic_slug == slug,
            CourseJob.status.in_(["pending", "running"]),
        )
    )
    if dup_job is not None:
        return CourseJobResponse(status="pending", job_id=dup_job.id)

    if search_token:
        course_preview.clear_preview(search_token)
    return _create_and_dispatch(db, user_id, topic_raw, canonical, embedding,
                                allow_duplicate=True)


def _create_and_dispatch(db: Session, user_id: int, topic_raw: str,
                         canonical: str, embedding: list[float],
                         allow_duplicate: bool = False) -> CourseJobResponse:
    job = CourseJob(
        topic_slug=_slugify(canonical), topic_raw=topic_raw, topic_embedding=embedding,
        status="pending", created_by_user_id=user_id,
        allow_duplicate=allow_duplicate,
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        # Mirror register()'s belt-and-suspenders: two concurrent requests for
        # the same canonical topic can both pass the dedup checks before either
        # commits, so the partial unique index on active (topic_slug) is the
        # real source of truth. The loser rolls back and attaches to the active
        # job (or finished course) that won the race, matched by the exact slug
        # the index is keyed on.
        db.rollback()
        dup = db.scalar(
            select(CourseJob).where(
                CourseJob.topic_slug == job.topic_slug,
                CourseJob.status.in_(["pending", "running"]),
            )
        )
        if dup is not None:
            return CourseJobResponse(status="pending", job_id=dup.id)
        course = db.scalar(select(Course).where(Course.topic_slug == job.topic_slug))
        if course is not None:
            return CourseJobResponse(status="exists", course=serialize_course(db, course, user_id))
        raise HTTPException(status_code=500, detail="could not enqueue course job")
    db.refresh(job)
    # pyright sees the plain function signature behind the @celery_app.task
    # decorator rather than the Task object it's replaced with at runtime,
    # so it doesn't know about `.delay` here -- same root cause as the
    # bind=True mismatch worked around in test_course_creation_task.py.
    run_course_creation_job.delay(job.id)  # pyright: ignore[reportFunctionMemberAccess]
    return CourseJobResponse(status="pending", job_id=job.id)


def get_course_job(db: Session, job_id: int, user_id: int | None = None) -> CourseJobResponse:
    job = db.get(CourseJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    course = None
    if job.status == "succeeded" and job.course_id:
        course_row = db.get(Course, job.course_id)
        if course_row is not None:
            course = serialize_course(db, course_row, user_id)
    return CourseJobResponse(status=job.status, job_id=job.id, course=course, error=job.error)


# How long a finished (succeeded/failed) job keeps showing up in
# list_my_course_jobs after it stops being pending/running — long enough for
# a learner who stepped away mid-generation to come back and see the result,
# short enough that the list doesn't accumulate months of old noise.
RECENTLY_FINISHED_WINDOW = timedelta(minutes=15)


def list_my_course_jobs(db: Session, user_id: int) -> list[MyCourseJobResponse]:
    """Jobs this user personally triggered: always includes in-flight ones,
    plus ones that finished (succeeded/failed) recently. Excludes jobs other
    users triggered, even if this user later attached to one via dedup — that
    job still shows up for the caller via its `job_id` (tracked client-side),
    just not in this "other jobs of mine" list."""
    cutoff = datetime.now(timezone.utc) - RECENTLY_FINISHED_WINDOW
    jobs = db.scalars(
        select(CourseJob)
        .where(
            CourseJob.created_by_user_id == user_id,
            or_(
                CourseJob.status.in_(["pending", "running"]),
                CourseJob.updated_at >= cutoff,
            ),
        )
        .order_by(CourseJob.created_at.desc())
    ).all()
    result = []
    for job in jobs:
        course_slug = None
        if job.status == "succeeded" and job.course_id:
            course_row = db.get(Course, job.course_id)
            course_slug = course_row.topic_slug if course_row is not None else None
        result.append(MyCourseJobResponse(
            id=job.id, topic_slug=job.topic_slug, topic_raw=job.topic_raw,
            status=job.status, error=job.error, course_slug=course_slug, created_at=job.created_at,
        ))
    return result


def retry_course_job(db: Session, job_id: int, user_id: int | None = None) -> CourseJobResponse:
    """Re-run a failed course job so no job is dead-ended. Before regenerating,
    dedup is re-checked on the job's stored canonical embedding: if a course for
    the topic exists now it is attached instead of paid for again; if another
    job is in flight, that one is returned."""
    job = db.get(CourseJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == "succeeded":
        course = db.get(Course, job.course_id) if job.course_id else None
        return CourseJobResponse(
            status="exists", job_id=job.id,
            course=serialize_course(db, course, user_id) if course else None,
        )
    if job.status != "failed":
        raise HTTPException(status_code=409, detail="only failed jobs can be retried")

    found = find_existing(job.topic_embedding, db)
    if isinstance(found, Course):
        course = _db_course(db, found)
        if course is not None:
            job.status = "succeeded"
            job.course_id = course.id
            job.error = None
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
            return CourseJobResponse(status="exists", job_id=job.id,
                                     course=serialize_course(db, course, user_id))
    if isinstance(found, CourseJob):
        # Another job already owns this topic; point the caller at it.
        return CourseJobResponse(status="pending", job_id=found.id)

    job.status = "pending"
    job.error = None
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    # A human-triggered retry must regenerate fresh, not blindly re-drive the
    # node that failed last time — drop the interrupted run's checkpoints so
    # the next invocation starts from scratch.
    purge_checkpoints(str(job_id))
    # Same .delay blind-spot as create_course above.
    run_course_creation_job.delay(job.id)  # pyright: ignore[reportFunctionMemberAccess]
    return CourseJobResponse(status="pending", job_id=job.id)


def list_public_courses(
    db: Session, user_id: int, *,
    search: str | None = None,
    sort: str = "date",
    order: str = "desc",
    page: int = 1,
    limit: int = 20,
) -> tuple[list[PublicCourseResponse], int]:
    tracked_ids = select(UserCourse.course_id).where(UserCourse.user_id == user_id)
    query = select(Course).where(Course.id.notin_(tracked_ids))
    if search:
        query = query.where(Course.topic_raw.ilike(f"%{search}%"))

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0

    sort_column = Course.topic_raw if sort == "name" else Course.created_at
    query = query.order_by(sort_column.asc() if order == "asc" else sort_column.desc())
    query = query.offset((page - 1) * limit).limit(limit)

    result = []
    for course in db.scalars(query).all():
        modules = db.query(Module).filter(*visible_module_filter(course.id, None)).all()
        chapter_count = sum(
            db.query(Chapter).filter_by(module_id=m.id).count() for m in modules
        )
        result.append(PublicCourseResponse(
            id=course.id, topic_slug=course.topic_slug, topic_raw=course.topic_raw,
            created_at=course.created_at,
            module_count=len(modules), chapter_count=chapter_count,
        ))
    return result, total
