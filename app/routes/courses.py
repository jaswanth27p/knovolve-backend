import json
import logging
import time
from datetime import datetime, timezone
from typing import Iterator
import redis
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import inspect as sa_inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.config import settings
from app.db import get_session
from app.enroll import touch_enrollment
from app.auth.dependencies import get_current_user
from app.models.course import Course, CourseJob, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.enrollment import UserCourse
from app.models.assignment import Assignment, AssignmentQuestion
from app.schemas.course import CreateCourseRequest, CourseJobResponse, PublicCourseResponse
from app.schemas.assignment import AssignmentQuestionResponse, AssignmentResponse
from app.agents.course_creation.nodes.normalize_topic import (
    _canonicalize,
    _slugify,
    find_existing,
)
from app.llm.factory import embed
from app.agents.course_creation.checkpoints import purge_checkpoints
from app.agents.chapter_content.generate import stream_chapter_content
from app.realtime.chapter_content_events import channel_name
from app.tasks.course_creation_task import run_course_creation_job
from app.tasks.assignment_tasks import generate_chapter_assignment_task, generate_module_assignment_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/courses", tags=["courses"])


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


def _serialize_course(course: Course, db: Session) -> dict:
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


def _existing_response(existing: Course | CourseJob, response: Response,
                       db: Session) -> CourseJobResponse:
    """Dedup hit -> response with the correct HTTP status: a completed read
    (course found) is 200; an in-flight duplicate job is 202."""
    if isinstance(existing, Course):
        response.status_code = 200
        course = _db_course(db, existing)
        if course is None:
            raise HTTPException(status_code=500, detail="existing course not found")
        return CourseJobResponse(
            status="exists", job_id=course.id, course=_serialize_course(course, db)
        )
    return CourseJobResponse(status="pending", job_id=existing.id)


def _canonical_topic(raw: str) -> str:
    """Canonicalize the topic; an LLM flap must not 500 the request, so fall
    back to the raw string (degraded dedup, still functional)."""
    try:
        return _canonicalize(raw)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to raw
        logger.warning("topic canonicalization failed; falling back to raw: %s", exc)
        return raw


@router.post("", response_model=CourseJobResponse, status_code=202)
def create_course(body: CreateCourseRequest, response: Response, db: Session = Depends(get_session),
                    user=Depends(get_current_user)):
    # Canonicalize + embed ONCE, and dedup on the canonical embedding so
    # semantically-similar-but-differently-worded topics collide correctly.
    canonical = _canonical_topic(body.topic)
    embedding = embed(canonical)

    existing = find_existing(embedding, db)
    if existing is not None:
        if isinstance(existing, Course):
            touch_enrollment(db, user.id, _db_course(db, existing))
        return _existing_response(existing, response, db)

    job = CourseJob(
        topic_slug=_slugify(canonical), topic_raw=body.topic, topic_embedding=embedding,
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
            response.status_code = 200
            return CourseJobResponse(status="exists", course=_serialize_course(course, db))
        raise HTTPException(status_code=500, detail="could not enqueue course job")
    db.refresh(job)
    # pyright sees the plain function signature behind the @celery_app.task
    # decorator rather than the Task object it's replaced with at runtime,
    # so it doesn't know about `.delay` here -- same root cause as the
    # bind=True mismatch worked around in test_course_creation_task.py.
    run_course_creation_job.delay(job.id)  # pyright: ignore[reportFunctionMemberAccess]
    return CourseJobResponse(status="pending", job_id=job.id)

@router.get("/jobs/{job_id}", response_model=CourseJobResponse)
def get_job(job_id: int, db: Session = Depends(get_session), user=Depends(get_current_user)):
    job = db.get(CourseJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    course = None
    if job.status == "succeeded" and job.course_id:
        course_row = db.get(Course, job.course_id)
        if course_row is not None:
            course = _serialize_course(course_row, db)
    return CourseJobResponse(status=job.status, job_id=job.id, course=course, error=job.error)

@router.post("/jobs/{job_id}/retry", response_model=CourseJobResponse, status_code=202)
def retry_job(job_id: int, response: Response, db: Session = Depends(get_session),
              user=Depends(get_current_user)):
    """Re-run a failed course job so no job is dead-ended. Before regenerating,
    dedup is re-checked on the job's stored canonical embedding: if a course for
    the topic exists now it is attached instead of paid for again; if another
    job is in flight, that one is returned."""
    job = db.get(CourseJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == "succeeded":
        response.status_code = 200
        course = db.get(Course, job.course_id) if job.course_id else None
        return CourseJobResponse(
            status="exists", job_id=job.id,
            course=_serialize_course(course, db) if course else None,
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
            response.status_code = 200
            return CourseJobResponse(status="exists", job_id=job.id,
                                     course=_serialize_course(course, db))
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

@router.get("", response_model=list[PublicCourseResponse])
def list_public_courses(db: Session = Depends(get_session), user=Depends(get_current_user)):
    tracked_ids = set(
        row.course_id for row in db.query(UserCourse).filter_by(user_id=user.id).all()
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


@router.get("/{slug}")
def get_course(slug: str, db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    touch_enrollment(db, user.id, course)
    return _serialize_course(course, db)


def _tail_pending_diagrams(chapter_content_id: int, pending_orders: set[int], db: Session) -> Iterator[str]:
    # Subscribe BEFORE draining/scanning. Redis pub/sub delivers only to live
    # subscribers, and the render task commits Postgres *before* it publishes, so:
    #   - anything committed before our SUBSCRIBE is invisible to the channel but
    #     caught by the periodic DB reconcile below;
    #   - anything committed after our SUBSCRIBE is delivered to us on the channel.
    # The old order (reconcile-then-subscribe) left a window — a diagram finishing
    # between the two was missed by both paths and stuck "pending" forever.
    client = redis.Redis.from_url(settings.redis_url)
    pubsub = client.pubsub()
    pubsub.subscribe(channel_name(chapter_content_id))
    deadline = time.monotonic() + settings.diagram_stream_timeout_seconds
    try:
        while pending_orders:
            resolved = _reconcile_pending_diagrams(chapter_content_id, pending_orders, db)
            for event in resolved:
                yield json.dumps(event) + "\n"
            if not pending_orders:
                break
            if time.monotonic() >= deadline:
                break
            message = pubsub.get_message(timeout=min(1.0, deadline - time.monotonic()))
            if message is None or message["type"] != "message":
                continue
            event = json.loads(message["data"])
            order = event.get("order")
            if order in pending_orders and event["type"] in ("diagram_ready", "diagram_failed"):
                yield json.dumps(event) + "\n"
                pending_orders.discard(order)
    finally:
        pubsub.close()

    # Final chance to reconcile (e.g. a diagram that resolved just as the deadline
    # hit and whose publish happened before our subscribe). Cheap, idempotent.
    for event in _reconcile_pending_diagrams(chapter_content_id, pending_orders, db):
        yield json.dumps(event) + "\n"


def _reconcile_pending_diagrams(chapter_content_id: int, pending_orders: set[int],
                                db: Session) -> list[dict]:
    """Resolve any pending diagram orders whose DB status has since moved to
    ready/failed (their publish may predate our subscribe). Returns the events
    to emit and mutates `pending_orders` to remove resolved orders."""
    events: list[dict] = []
    sections = db.scalars(
        select(ChapterContentSection).where(
            ChapterContentSection.chapter_content_id == chapter_content_id,
            ChapterContentSection.order.in_(pending_orders),
        )
    )
    for section in sections:
        if section.diagram_status == "ready":
            events.append({"type": "diagram_ready", "order": section.order,
                           "diagram_image_url": section.diagram_image_url})
            pending_orders.discard(section.order)
        elif section.diagram_status == "failed":
            events.append({"type": "diagram_failed", "order": section.order})
            pending_orders.discard(section.order)
    return events


@router.get("/{slug}/chapters/{chapter_id}/content")
def get_chapter_content(slug: str, chapter_id: int, db: Session = Depends(get_session),
                          user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    chapter = (
        db.query(Chapter)
        .join(Module, Chapter.module_id == Module.id)
        .filter(Chapter.id == chapter_id, Module.course_id == course.id)
        .first()
    )
    if not chapter:
        raise HTTPException(status_code=404, detail="chapter not found")
    touch_enrollment(db, user.id, course)

    def _generate() -> Iterator[str]:
        pending_orders: set[int] = set()
        terminal: dict | None = None
        for event in stream_chapter_content(chapter, db):
            if event["type"] in ("done", "error"):
                # Hold the terminal event until after any pending-diagram tail so
                # the stream emits exactly one terminal ("done"|"error").
                terminal = event
                break
            yield json.dumps(event) + "\n"
            if event["type"] == "section_ready" and event["diagram_status"] == "pending":
                pending_orders.add(event["order"])

        if pending_orders and terminal and terminal["type"] == "done":
            # Only tail on a clean finish. If generation errored, the chapter is
            # failed and any earlier sections' diagrams are best resolved on the
            # next open's DB replay — blocking up to the timeout to hear about
            # them would only delay surfacing the error to the user.
            content = db.scalar(
                select(ChapterContent).where(ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global")
            )
            if content is not None:
                yield from _tail_pending_diagrams(content.id, pending_orders, db)

        yield json.dumps(terminal or {"type": "done"}) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")


def _serialize_assignment(assignment: Assignment, db: Session) -> AssignmentResponse:
    if assignment.status != "ready":
        return AssignmentResponse(status=assignment.status, error=assignment.error)
    questions = (
        db.query(AssignmentQuestion)
        .filter_by(assignment_id=assignment.id)
        .order_by(AssignmentQuestion.order)
        .all()
    )
    return AssignmentResponse(
        status="ready",
        questions=[
            AssignmentQuestionResponse(
                id=q.id, order=q.order, type=q.type, text=q.text, options=q.options,
                correct_answer=q.correct_answer, explanation=q.explanation,
                concept_tag=q.concept_tag, difficulty=q.difficulty,
            )
            for q in questions
        ],
    )


@router.get("/{slug}/chapters/{chapter_id}/assignment", response_model=AssignmentResponse)
def get_chapter_assignment(slug: str, chapter_id: int, db: Session = Depends(get_session),
                            user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    chapter = (
        db.query(Chapter)
        .join(Module, Chapter.module_id == Module.id)
        .filter(Chapter.id == chapter_id, Module.course_id == course.id)
        .first()
    )
    if not chapter:
        raise HTTPException(status_code=404, detail="chapter not found")

    content = db.scalar(
        select(ChapterContent).where(ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global")
    )
    if content is None or content.status != "ready":
        raise HTTPException(status_code=404, detail="chapter content not ready")

    assignment = db.scalar(select(Assignment).where(Assignment.chapter_content_id == content.id))
    if assignment is None or assignment.status == "failed":
        generate_chapter_assignment_task.delay(content.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = db.scalar(select(Assignment).where(Assignment.chapter_content_id == content.id))
    if assignment is None:
        return AssignmentResponse(status="generating")
    return _serialize_assignment(assignment, db)


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


@router.post("/{slug}/modules/{module_id}/assignment", response_model=AssignmentResponse, status_code=202)
def create_module_assignment(slug: str, module_id: int, response: Response,
                              db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    module = db.query(Module).filter_by(id=module_id, course_id=course.id).first()
    if not module:
        raise HTTPException(status_code=404, detail="module not found")
    if not _module_chapters_ready(module, db):
        raise HTTPException(status_code=409, detail="not all chapters in this module have ready content")

    assignment = db.scalar(select(Assignment).where(Assignment.module_id == module.id))
    if assignment is None or assignment.status == "failed":
        generate_module_assignment_task.delay(module.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = db.scalar(select(Assignment).where(Assignment.module_id == module.id))
    if assignment is None:
        return AssignmentResponse(status="generating")
    response.status_code = 200 if assignment.status == "ready" else 202
    return _serialize_assignment(assignment, db)


@router.get("/{slug}/modules/{module_id}/assignment", response_model=AssignmentResponse)
def get_module_assignment(slug: str, module_id: int, db: Session = Depends(get_session),
                           user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    module = db.query(Module).filter_by(id=module_id, course_id=course.id).first()
    if not module:
        raise HTTPException(status_code=404, detail="module not found")

    assignment = db.scalar(select(Assignment).where(Assignment.module_id == module.id))
    if assignment is None or assignment.status == "failed":
        if not _module_chapters_ready(module, db):
            raise HTTPException(status_code=409, detail="not all chapters in this module have ready content")
        generate_module_assignment_task.delay(module.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = db.scalar(select(Assignment).where(Assignment.module_id == module.id))
    if assignment is None:
        return AssignmentResponse(status="generating")
    return _serialize_assignment(assignment, db)
