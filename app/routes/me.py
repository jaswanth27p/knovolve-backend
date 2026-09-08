from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.models.user import User
from app.schemas.course import DashboardResponse, TrackedCourseResponse
from app.services import tracking

router = APIRouter(prefix="/me", tags=["me"])


@router.get("/courses", response_model=list[TrackedCourseResponse])
def get_my_courses(db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    return tracking.list_my_courses(db, user.id)


@router.get("/dashboard", response_model=DashboardResponse)
def get_dashboard(db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    return tracking.get_dashboard(db, user.id)


@router.delete("/courses/{course_id}", status_code=204)
def delete_my_course(course_id: int, db: Session = Depends(get_session),
                     user: User = Depends(get_current_user)):
    tracking.untrack_course(db, user.id, course_id)
    return Response(status_code=204)
