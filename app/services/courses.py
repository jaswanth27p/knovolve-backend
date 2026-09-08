"""Course catalog service: course generation jobs, dedup, list/fetch courses."""
import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import inspect as sa_inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.course_creation.checkpoints import purge_checkpoints
from app.agents.course_creation.nodes.normalize_topic import (
    _canonicalize,
    _slugify,
    find_existing,
)
from app.llm.factory import embed
from app.models.course import Course, CourseJob, Module, Chapter
from app.models.enrollment import UserCourse
from app.schemas.course import CourseJobResponse, PublicCourseResponse
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


def serialize_course(db: Session, course: Course) -> dict:
    modules = db.query(Module).filter_by(course_id=course.id).order_by(Module.order).all()
    return {
        "id": course.id, "topic_slug": course.topic_slug, "topic_raw": course.topic_raw,
        "modules": [
            {"id": m.id, "title": m.title, "objective": m.objective,
             "chapters": [{"id": c.id, "title": c.title, "objective": c.objective}
                          for c in db.query(Chapter).filter_by(module_id=m.id).order_by(Chapter.order).all()]}
            for m in modules
        ],
    }


def _canonical_topic(raw: str) -> str:
    """Canonicalize the topic; an LLM flap must not 500 the request, so fall
    back to the raw string (degraded dedup, still functional)."""
    try:
        return _canonicalize(raw)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to raw
        logger.warning("topic canonicalization failed; falling back to raw: %s", exc)
        return raw


def _existing_response(existing: Course | CourseJob, db: Session) -> CourseJobResponse:
    """Dedup hit -> response object. status="exists" (course found) or
    status="pending" (in-flight duplicate job); the route maps "exists" to HTTP
    200 and leaves the decorator's 202 default for "pending"."""
    if isinstance(existing, Course):
        course = _db_course(db, existing)
        if course is None:
            raise HTTPException(status_code=500, detail="existing course not found")
        return CourseJobResponse(
            status="exists", job_id=course.id, course=serialize_course(db, course)
        )
    return CourseJobResponse(status="pending", job_id=existing.id)


def get_course_by_slug(db: Session, slug: str) -> Course:
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    return course


def get_module(db: Session, course: Course, module_id: int) -> Module:
    module = db.query(Module).filter_by(id=module_id, course_id=course.id).first()
    if not module:
        raise HTTPException(status_code=404, detail="module not found")
    return module


def create_course_job(db: Session, user_id: int, topic_raw: str) -> CourseJobResponse:
    # Canonicalize + embed ONCE, and dedup on the canonical embedding so
    # semantically-similar-but-differently-worded topics collide correctly.
    canonical = _canonical_topic(topic_raw)
    embedding = embed(canonical)

    existing = find_existing(embedding, db)
    if existing is not None:
        if isinstance(existing, Course):
            touch_enrollment(db, user_id, _db_course(db, existing))
        return _existing_response(existing, db)

    job = CourseJob(
        topic_slug=_slugify(canonical), topic_raw=topic_raw, topic_embedding=embedding,
        status="pending", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError:
        # Mirror register()'s belt-and-suspenders: two concurrent requests for
        # the same canonical topic can both pass find_existing before either
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
            return CourseJobResponse(status="exists", course=serialize_course(db, course))
        raise HTTPException(status_code=500, detail="could not enqueue course job")
    db.refresh(job)
    # pyright sees the plain function signature behind the @celery_app.task
    # decorator rather than the Task object it's replaced with at runtime,
    # so it doesn't know about `.delay` here -- same root cause as the
    # bind=True mismatch worked around in test_course_creation_task.py.
    run_course_creation_job.delay(job.id)  # pyright: ignore[reportFunctionMemberAccess]
    return CourseJobResponse(status="pending", job_id=job.id)


def get_course_job(db: Session, job_id: int) -> CourseJobResponse:
    job = db.get(CourseJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    course = None
    if job.status == "succeeded" and job.course_id:
        course_row = db.get(Course, job.course_id)
        if course_row is not None:
            course = serialize_course(db, course_row)
    return CourseJobResponse(status=job.status, job_id=job.id, course=course, error=job.error)


def retry_course_job(db: Session, job_id: int) -> CourseJobResponse:
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
            course=serialize_course(db, course) if course else None,
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
                                     course=serialize_course(db, course))
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


def list_public_courses(db: Session, user_id: int) -> list[PublicCourseResponse]:
    tracked_ids = set(
        row.course_id for row in db.query(UserCourse).filter_by(user_id=user_id).all()
    )
    courses = db.query(Course).order_by(Course.created_at.desc()).all()
    result = []
    for course in courses:
        if course.id in tracked_ids:
            continue
        modules = db.query(Module).filter_by(course_id=course.id).all()
        chapter_count = sum(
            db.query(Chapter).filter_by(module_id=m.id).count() for m in modules
        )
        result.append(PublicCourseResponse(
            id=course.id, topic_slug=course.topic_slug, topic_raw=course.topic_raw,
            module_count=len(modules), chapter_count=chapter_count,
        ))
    return result
