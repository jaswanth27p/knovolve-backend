from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.course import DashboardResponse, TrackedCourseResponse
from app.services import tracking
from app.services import chat as chat_service

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


@router.post("/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest, db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    reply = chat_service.answer_chat_message(db, user.id, body.course_slug, body.chapter_id, body.message)
    return ChatResponse(reply=reply)
