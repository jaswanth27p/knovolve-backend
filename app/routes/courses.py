import json
from typing import Iterator
from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.db import get_session
from app.services import assignments, attempts, chapter_content, courses
from app.services.enrollment import touch_enrollment
from app.auth.dependencies import get_current_user
from app.schemas.course import CreateCourseRequest, CourseJobResponse, PublicCourseResponse
from app.schemas.assignment import AssignmentResponse
from app.schemas.attempt import SubmitAttemptRequest, AttemptSubmitResponse, AttemptResponse

router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("", response_model=CourseJobResponse, status_code=202)
def create_course(body: CreateCourseRequest, response: Response, db: Session = Depends(get_session),
                  user=Depends(get_current_user)):
    result = courses.create_course_job(db, user.id, body.topic)
    if result.status == "exists":
        response.status_code = 200
    return result


@router.get("/jobs/{job_id}", response_model=CourseJobResponse)
def get_job(job_id: int, db: Session = Depends(get_session), user=Depends(get_current_user)):
    return courses.get_course_job(db, job_id)


@router.post("/jobs/{job_id}/retry", response_model=CourseJobResponse, status_code=202)
def retry_job(job_id: int, response: Response, db: Session = Depends(get_session),
              user=Depends(get_current_user)):
    result = courses.retry_course_job(db, job_id)
    if result.status == "exists":
        response.status_code = 200
    return result


@router.get("", response_model=list[PublicCourseResponse])
def list_public_courses(db: Session = Depends(get_session), user=Depends(get_current_user)):
    return courses.list_public_courses(db, user.id)


@router.get("/{slug}")
def get_course(slug: str, db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    touch_enrollment(db, user.id, course)
    return courses.serialize_course(db, course)


@router.get("/{slug}/chapters/{chapter_id}/content")
def get_chapter_content(slug: str, chapter_id: int, db: Session = Depends(get_session),
                        user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    chapter = chapter_content.get_chapter(db, course, chapter_id)
    touch_enrollment(db, user.id, course)

    def _encode(events: Iterator[dict]) -> Iterator[str]:
        for event in events:
            yield json.dumps(event) + "\n"

    return StreamingResponse(
        _encode(chapter_content.stream_content_events(chapter, db)),
        media_type="application/x-ndjson",
    )


@router.get("/{slug}/chapters/{chapter_id}/assignment", response_model=AssignmentResponse)
def get_chapter_assignment(slug: str, chapter_id: int, db: Session = Depends(get_session),
                            user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    chapter = chapter_content.get_chapter(db, course, chapter_id)
    return assignments.get_chapter_assignment(db, chapter)


@router.post("/{slug}/modules/{module_id}/assignment", response_model=AssignmentResponse, status_code=202)
def create_module_assignment(slug: str, module_id: int, response: Response,
                              db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    module = courses.get_module(db, course, module_id)
    result = assignments.create_module_assignment(db, module, user.id)
    if result.status == "ready":
        response.status_code = 200
    return result


@router.get("/{slug}/modules/{module_id}/assignment", response_model=AssignmentResponse)
def get_module_assignment(slug: str, module_id: int, db: Session = Depends(get_session),
                           user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    module = courses.get_module(db, course, module_id)
    return assignments.get_module_assignment(db, module, user.id)


@router.post("/{slug}/assignments/{assignment_id}/attempts", response_model=AttemptSubmitResponse, status_code=202)
def submit_assignment_attempt(slug: str, assignment_id: int, body: SubmitAttemptRequest,
                               db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return attempts.submit_attempt(db, user.id, course, assignment_id, body)


@router.get("/{slug}/assignments/{assignment_id}/attempts/{attempt_id}", response_model=AttemptResponse)
def get_assignment_attempt(slug: str, assignment_id: int, attempt_id: int,
                            db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return attempts.get_attempt(db, user.id, course, assignment_id, attempt_id)
