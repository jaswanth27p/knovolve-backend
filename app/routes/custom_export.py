from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.models.user import User
from app.schemas.export import ClarifyRequest, ClarifyResponse
from app.services import courses, exports as export_service

router = APIRouter(prefix="/courses/{slug}", tags=["custom-export"])


@router.post("/export-clarify", response_model=ClarifyResponse)
def clarify_custom_export(slug: str, body: ClarifyRequest, db: Session = Depends(get_session),
                          user: User = Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    result = export_service.run_clarify(db, user.id, course, body.message, body.history)
    return ClarifyResponse(**result)
