from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db import get_session
from app.models.user import User
from app.schemas.course import (
    CreateExtensionRequest,
    ExtensionChapterSummary,
    ExtensionJobResponse,
)
from app.services import course_extension, courses

router = APIRouter(prefix="/courses/{slug}/extensions", tags=["courses"])


@router.post("", response_model=ExtensionJobResponse, status_code=202)
def create_extension(slug: str, body: CreateExtensionRequest, response: Response,
                     db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    job = course_extension.create_extension_job(db, user.id, course, body.message)
    return ExtensionJobResponse(status=job.status, job_id=job.id, request=job.request,
                                created_at=job.created_at, updated_at=job.updated_at)


@router.get("/jobs", response_model=list[ExtensionJobResponse])
def list_extension_jobs(slug: str, db: Session = Depends(get_session),
                        user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return [
        ExtensionJobResponse(
            status=job.status, job_id=job.id, error=job.error, added=job.result,
            request=job.request, created_at=job.created_at, updated_at=job.updated_at,
        )
        for job in course_extension.list_extension_jobs(db, user.id, course)
    ]


@router.get("/jobs/latest", response_model=ExtensionJobResponse | None)
def get_latest_extension_job(slug: str, db: Session = Depends(get_session),
                             user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    job = course_extension.get_latest_extension_job(db, user.id, course)
    if job is None:
        return None
    return ExtensionJobResponse(
        status=job.status, job_id=job.id, error=job.error, added=job.result,
        request=job.request, created_at=job.created_at, updated_at=job.updated_at,
    )


@router.get("/jobs/{job_id}", response_model=ExtensionJobResponse)
def get_extension_job(slug: str, job_id: int, db: Session = Depends(get_session),
                      user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    job = course_extension.get_extension_job(db, user.id, course, job_id)
    return ExtensionJobResponse(
        status=job.status, job_id=job.id, error=job.error, added=job.result,
        request=job.request, created_at=job.created_at, updated_at=job.updated_at,
    )


@router.get("/chapters", response_model=list[ExtensionChapterSummary])
def list_extension_chapters(slug: str, db: Session = Depends(get_session),
                            user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return [ExtensionChapterSummary(**c) for c in course_extension.list_extension_chapters(db, user.id, course)]


@router.delete("/chapters/{chapter_id}", status_code=204)
def delete_extension_chapter(slug: str, chapter_id: int, db: Session = Depends(get_session),
                             user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    course_extension.delete_extension_chapter(db, user.id, course, chapter_id)
    return Response(status_code=204)