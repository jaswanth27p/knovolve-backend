from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.models.export import ExportJob
from app.models.user import User
from app.schemas.export import CreateExportRequest, ExportJobResponse
from app.services import courses, exports as export_service
from app.tasks.export_tasks import run_export_task

router = APIRouter(prefix="/courses/{slug}/exports", tags=["exports"])


def _serialize(job: ExportJob) -> ExportJobResponse:
    return ExportJobResponse(
        id=job.id,
        kind=job.kind,
        status=job.status,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
        result_size=job.result_size,
    )


@router.post("", response_model=ExportJobResponse, status_code=202)
def create_export(slug: str, body: CreateExportRequest, db: Session = Depends(get_session),
                  user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    job = export_service.create_export_job(db, user.id, course, body.kind, body.params)
    run_export_task.delay(job.id)  # pyright: ignore[reportFunctionMemberAccess]
    return _serialize(job)


@router.get("", response_model=list[ExportJobResponse])
def list_exports(slug: str, db: Session = Depends(get_session),
                 user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return [_serialize(job) for job in export_service.list_export_jobs(db, user.id, course)]


@router.get("/{export_id}", response_model=ExportJobResponse)
def get_export(slug: str, export_id: int, db: Session = Depends(get_session),
               user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return _serialize(export_service.get_export_job(db, user.id, course, export_id))


@router.get("/{export_id}/download")
def download_export(slug: str, export_id: int, db: Session = Depends(get_session),
                    user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    url = export_service.presigned_export_url(db, user.id, course, export_id)
    return RedirectResponse(url=url, status_code=302)
