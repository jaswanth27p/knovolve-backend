import json
import math
from typing import Iterator, Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.course import DashboardResponse, PaginatedTrackedCoursesResponse
from app.services import tracking
from app.services import chat as chat_service

router = APIRouter(prefix="/me", tags=["me"])


@router.get("/courses", response_model=PaginatedTrackedCoursesResponse)
def get_my_courses(
    search: str | None = None,
    status: Literal["in_progress", "completed"] | None = None,
    sort: Literal["name", "date", "progress"] = "date",
    order: Literal["asc", "desc"] = "desc",
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_session), user: User = Depends(get_current_user),
):
    items, total = tracking.list_my_courses(
        db, user.id, search=search, status=status, sort=sort, order=order, page=page, limit=limit
    )
    return PaginatedTrackedCoursesResponse(
        items=items, total=total, page=page, limit=limit,
        total_pages=math.ceil(total / limit) if total else 0,
    )


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


@router.post("/chat/stream")
def post_chat_stream(body: ChatRequest, db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    def _encode(events: Iterator[dict]) -> Iterator[str]:
        for event in events:
            yield json.dumps(event) + "\n"

    return StreamingResponse(
        _encode(chat_service.stream_chat_message(db, user.id, body.course_slug, body.chapter_id, body.message)),
        media_type="application/x-ndjson",
    )
