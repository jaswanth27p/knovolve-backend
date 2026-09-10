import json
import math
from typing import Iterator, Literal
from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.db import get_session
from app.services import assignments, attempts, chapter_content, courses
from app.services.enrollment import touch_enrollment
from app.auth.dependencies import get_current_user
from app.schemas.course import (
    ChapterVersionDetail,
    ChapterVersionSummary,
    CreateCourseRequest,
    CourseJobResponse,
    MyCourseJobResponse,
    PaginatedPublicCoursesResponse,
    PublicCourseResponse,
)
from app.schemas.assignment import AssignmentResponse
from app.schemas.attempt import SubmitAttemptRequest, AttemptSubmitResponse, AttemptResponse, AttemptSummary

router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("", response_model=CourseJobResponse, status_code=202)
def create_course(body: CreateCourseRequest, response: Response, db: Session = Depends(get_session),
                  user=Depends(get_current_user)):
    result = courses.create_course_job(db, user.id, body.topic,
                                       force=body.force, search_token=body.search_token)
    if result.status in ("exists", "similar"):
        response.status_code = 200
    return result


@router.get("/jobs/{job_id}", response_model=CourseJobResponse)
def get_job(job_id: int, db: Session = Depends(get_session), user=Depends(get_current_user)):
    return courses.get_course_job(db, job_id, user.id)


@router.get("/jobs", response_model=list[MyCourseJobResponse])
def list_my_jobs(db: Session = Depends(get_session), user=Depends(get_current_user)):
    return courses.list_my_course_jobs(db, user.id)


@router.post("/jobs/{job_id}/retry", response_model=CourseJobResponse, status_code=202)
def retry_job(job_id: int, response: Response, db: Session = Depends(get_session),
              user=Depends(get_current_user)):
    result = courses.retry_course_job(db, job_id, user.id)
    if result.status == "exists":
        response.status_code = 200
    return result


@router.get("", response_model=PaginatedPublicCoursesResponse)
def list_public_courses(
    search: str | None = None,
    sort: Literal["name", "date"] = "date",
    order: Literal["asc", "desc"] = "desc",
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_session), user=Depends(get_current_user),
):
    items, total = courses.list_public_courses(
        db, user.id, search=search, sort=sort, order=order, page=page, limit=limit
    )
    return PaginatedPublicCoursesResponse(
        items=items, total=total, page=page, limit=limit,
        total_pages=math.ceil(total / limit) if total else 0,
    )


@router.get("/{slug}")
def get_course(slug: str, db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    touch_enrollment(db, user.id, course)
    return courses.serialize_course(db, course, user.id)


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
        _encode(chapter_content.stream_content_events(chapter, db, user.id)),
        media_type="application/x-ndjson",
    )


@router.get("/{slug}/chapters/{chapter_id}/versions", response_model=list[ChapterVersionSummary])
def list_chapter_versions(slug: str, chapter_id: int, db: Session = Depends(get_session),
                          user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    chapter = chapter_content.get_chapter(db, course, chapter_id)
    return chapter_content.list_chapter_versions(db, chapter, user.id)


@router.get("/{slug}/chapters/{chapter_id}/versions/{version}", response_model=ChapterVersionDetail)
def get_chapter_version(slug: str, chapter_id: int, version: int, db: Session = Depends(get_session),
                        user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    chapter = chapter_content.get_chapter(db, course, chapter_id)
    return chapter_content.get_chapter_version(db, chapter, user.id, version)


@router.get("/{slug}/chapters/{chapter_id}/assignment", response_model=AssignmentResponse)
def get_chapter_assignment(slug: str, chapter_id: int, db: Session = Depends(get_session),
                            user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    chapter = chapter_content.get_chapter(db, course, chapter_id)
    return assignments.get_chapter_assignment(db, chapter, user.id)


@router.get("/{slug}/chapters/{chapter_id}/versions/{version}/assignment", response_model=AssignmentResponse)
def get_chapter_version_assignment(slug: str, chapter_id: int, version: int, db: Session = Depends(get_session),
                                    user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    chapter = chapter_content.get_chapter(db, course, chapter_id)
    return assignments.get_chapter_assignment_for_version(db, chapter, user.id, version)


@router.post("/{slug}/modules/{module_id}/assignment", response_model=AssignmentResponse, status_code=202)
def create_module_assignment(slug: str, module_id: int, response: Response,
                              db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    module = courses.get_module(db, course, module_id, user.id)
    result = assignments.create_module_assignment(db, module, user.id)
    if result.status == "ready":
        response.status_code = 200
    return result


@router.get("/{slug}/modules/{module_id}/assignment", response_model=AssignmentResponse)
def get_module_assignment(slug: str, module_id: int, db: Session = Depends(get_session),
                           user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    module = courses.get_module(db, course, module_id, user.id)
    return assignments.get_module_assignment(db, module, user.id)


@router.post("/{slug}/assignments/{assignment_id}/attempts", response_model=AttemptSubmitResponse, status_code=202)
def submit_assignment_attempt(slug: str, assignment_id: int, body: SubmitAttemptRequest,
                               db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return attempts.submit_attempt(db, user.id, course, assignment_id, body)


@router.get("/{slug}/assignments/{assignment_id}/attempts", response_model=list[AttemptSummary])
def list_assignment_attempts(slug: str, assignment_id: int, db: Session = Depends(get_session),
                             user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return attempts.list_attempts(db, user.id, course, assignment_id)


@router.get("/{slug}/assignments/{assignment_id}/attempts/{attempt_id}", response_model=AttemptResponse)
def get_assignment_attempt(slug: str, assignment_id: int, attempt_id: int,
                            db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return attempts.get_attempt(db, user.id, course, assignment_id, attempt_id)
