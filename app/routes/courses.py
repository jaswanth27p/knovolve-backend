from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.db import get_session
from app.auth.dependencies import get_current_user
from app.models.course import Course, CourseJob, Module, Chapter
from app.schemas.course import CreateCourseRequest, CourseJobResponse
from app.agents.course_creation.nodes.normalize_topic import find_existing, _slugify
from app.llm.factory import embed
from app.tasks.course_creation_task import run_course_creation_job

router = APIRouter(prefix="/courses", tags=["courses"])

def _serialize_course(course: Course, db: Session) -> dict:
    modules = db.query(Module).filter_by(course_id=course.id).order_by(Module.order).all()
    return {
        "id": course.id, "topic_slug": course.topic_slug, "topic_raw": course.topic_raw,
        "modules": [
            {"title": m.title, "objective": m.objective,
             "chapters": [{"title": c.title, "objective": c.objective}
                          for c in db.query(Chapter).filter_by(module_id=m.id).order_by(Chapter.order).all()]}
            for m in modules
        ],
    }

@router.post("", status_code=202, response_model=CourseJobResponse)
def create_course(body: CreateCourseRequest, db: Session = Depends(get_session),
                    user=Depends(get_current_user)):
    existing = find_existing(body.topic, db)
    if isinstance(existing, Course):
        return CourseJobResponse(status="exists", course=_serialize_course(existing, db))
    if isinstance(existing, CourseJob):
        return CourseJobResponse(status="pending", job_id=existing.id)

    job = CourseJob(
        topic_slug=_slugify(body.topic), topic_raw=body.topic, topic_embedding=embed(body.topic),
        status="pending", created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()
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

@router.get("/{slug}")
def get_course(slug: str, db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    return _serialize_course(course, db)
