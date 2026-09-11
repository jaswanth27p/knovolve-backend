from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.models.export import CourseGenerationRun
from app.models.user import User
from app.schemas.export import CourseReadinessResponse, GenerationRunResponse
from app.services import courses, generation as generation_service
from app.tasks.course_generation_task import run_course_generation_task

router = APIRouter(prefix="/courses/{slug}", tags=["generation"])


def _serialize_run(run: CourseGenerationRun, action: str | None = None) -> GenerationRunResponse:
    return GenerationRunResponse(
        id=run.id,
        status=run.status,
        total_units=run.total_units,
        completed_units=run.completed_units,
        unit_states=run.unit_states or [],
        error=run.error,
        action=action,
    )


@router.post("/generate", response_model=GenerationRunResponse, status_code=202)
def queue_generation(slug: str, response: Response, db: Session = Depends(get_session),
                     user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    run, action = generation_service.queue_generation_run(db, user.id, course)
    if action == "queued":
        run_course_generation_task.delay(run.id)  # pyright: ignore[reportFunctionMemberAccess]
    else:
        response.status_code = 200
    return _serialize_run(run, action)


@router.get("/generation", response_model=GenerationRunResponse)
def get_generation(slug: str, db: Session = Depends(get_session),
                   user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    run = generation_service.get_latest_run(db, user.id, course)
    if run is None:
        raise HTTPException(status_code=404, detail="generation run not found")
    return _serialize_run(run)


@router.get("/readiness", response_model=CourseReadinessResponse)
def get_readiness(slug: str, db: Session = Depends(get_session),
                  user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    return CourseReadinessResponse(**generation_service.readiness_summary(db, course, user.id))
