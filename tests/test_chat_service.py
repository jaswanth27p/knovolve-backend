from unittest.mock import patch
from app.db import SessionLocal
from app.schemas.course import DashboardResponse, StreakResponse
from app.services import chat


def test_progress_question_routes_to_progress_tool():
    with patch("app.services.chat.get_dashboard", return_value=DashboardResponse(
        in_progress=[], completed=[], in_progress_count=1, completed_count=0, total_count=1,
        streak=StreakResponse(current=0, longest=0),
    )):
        with SessionLocal() as db:
            answer = chat.answer_chat_message(db, user_id=1, course_slug=None, chapter_id=None,
                                                message="what's my progress")
    assert "1" in answer


def test_result_question_without_chapter_context_asks_for_context():
    with SessionLocal() as db:
        answer = chat.answer_chat_message(db, user_id=1, course_slug="x", chapter_id=None,
                                            message="why did I get that wrong")
    assert "open a chapter" in answer.lower() or "which chapter" in answer.lower()
