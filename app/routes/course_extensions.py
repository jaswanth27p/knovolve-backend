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
from app.tasks.course_extension_task import run_course_extension_job

router = APIRouter(prefix="/courses/{slug}/extensions", tags=["courses"])


@router.post("", response_model=ExtensionJobResponse, status_code=202)
def create_extension(slug: str, body: CreateExtensionRequest, response: Response,
                     db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    job = course_extension.create_extension_job(db, user.id, course, body.message)
    run_course_extension_job.delay(job.id)  # pyright: ignore[reportFunctionMemberAccess]
    return ExtensionJobResponse(status=job.status, job_id=job.id)


@router.get("/jobs/{job_id}", response_model=ExtensionJobResponse)
def get_extension_job(slug: str, job_id: int, db: Session = Depends(get_session),
                      user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    job = course_extension.get_extension_job(db, user.id, course, job_id)
    return ExtensionJobResponse(
        status=job.status, job_id=job.id, error=job.error, added=job.result,
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