from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.db import SessionLocal
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Course, Module
from app.models.user import User
from app.schemas.course import DashboardResponse, StreakResponse
from app.services import chat


def _now():
    return datetime.now(timezone.utc)


def _make_chapter_with_assignment(db, slug: str):
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
    db.add(course); db.flush()
    module = Module(course_id=course.id, title="M", objective="o", order=1)
    db.add(module); db.flush()
    chapter = Chapter(module_id=module.id, title="C1", objective="o", order=1)
    db.add(chapter); db.flush()
    content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                              outline=[], created_at=_now(), updated_at=_now())
    db.add(content); db.flush()
    assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                             status="ready", created_at=_now(), updated_at=_now())
    db.add(assignment); db.flush()
    return course, chapter, assignment


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


def test_result_question_with_no_attempt_yet():
    with SessionLocal() as db:
        _, chapter, _ = _make_chapter_with_assignment(db, "chat-result-a")
        db.commit()
        chapter_id = chapter.id

    with SessionLocal() as db:
        answer = chat.answer_chat_message(db, user_id=999, course_slug="chat-result-a", chapter_id=chapter_id,
                                            message="what did I get wrong")
    assert "haven't completed" in answer.lower()


def test_result_question_reports_latest_graded_attempt():
    with SessionLocal() as db:
        _, chapter, assignment = _make_chapter_with_assignment(db, "chat-result-b")
        user = User(email="chat-result-b@example.com", password_hash="x")
        db.add(user); db.flush()
        db.add(AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                  overall_score=0.85, created_at=_now(), updated_at=_now()))
        db.commit()
        chapter_id, user_id = chapter.id, user.id

    with SessionLocal() as db:
        answer = chat.answer_chat_message(db, user_id=user_id, course_slug="chat-result-b", chapter_id=chapter_id,
                                            message="what was my score")
    assert "85%" in answer
    assert "passed" in answer.lower()


def test_freeform_question_calls_llm_grounded_in_context():
    """A message that matches none of the fixed keyword shapes must not fall
    back to a canned string — it goes to the LLM with real learner context."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content="You're doing great, keep going!")
    with patch("app.services.chat.get_dashboard", return_value=DashboardResponse(
        in_progress=[], completed=[], in_progress_count=2, completed_count=1, total_count=3,
        streak=StreakResponse(current=3, longest=5),
    )), patch("app.services.chat.get_chat_model", return_value=mock_model):
        with SessionLocal() as db:
            answer = chat.answer_chat_message(db, user_id=1, course_slug=None, chapter_id=None,
                                                message="any study tips for today?")

    assert answer == "You're doing great, keep going!"
    mock_model.invoke.assert_called_once()
    # The prompt sent to the LLM must be grounded in the real dashboard data,
    # not a hardcoded placeholder.
    messages = mock_model.invoke.call_args.args[0]
    system_content = messages[0].content
    assert "3 day(s)" in system_content


def test_stream_deterministic_reply_yields_single_token_then_done():
    with patch("app.services.chat.get_dashboard", return_value=DashboardResponse(
        in_progress=[], completed=[], in_progress_count=1, completed_count=0, total_count=1,
        streak=StreakResponse(current=0, longest=0),
    )):
        with SessionLocal() as db:
            events = list(chat.stream_chat_message(db, user_id=1, course_slug=None, chapter_id=None,
                                                     message="what's my progress"))
    assert events[-1] == {"type": "done"}
    assert events[0]["type"] == "token"
    assert "1" in events[0]["text"]


def test_stream_freeform_reply_yields_one_token_event_per_chunk():
    mock_model = MagicMock()
    mock_model.stream.return_value = [MagicMock(content="You're "), MagicMock(content="doing great!")]
    with patch("app.services.chat.get_dashboard", return_value=DashboardResponse(
        in_progress=[], completed=[], in_progress_count=2, completed_count=1, total_count=3,
        streak=StreakResponse(current=3, longest=5),
    )), patch("app.services.chat.get_chat_model", return_value=mock_model):
        with SessionLocal() as db:
            events = list(chat.stream_chat_message(db, user_id=1, course_slug=None, chapter_id=None,
                                                     message="any study tips for today?"))
    assert events == [
        {"type": "token", "text": "You're "},
        {"type": "token", "text": "doing great!"},
        {"type": "done"},
    ]


def test_stream_freeform_reply_yields_error_event_on_llm_failure():
    mock_model = MagicMock()
    mock_model.stream.side_effect = RuntimeError("boom")
    with patch("app.services.chat.get_dashboard", return_value=DashboardResponse(
        in_progress=[], completed=[], in_progress_count=0, completed_count=0, total_count=0,
        streak=StreakResponse(current=0, longest=0),
    )), patch("app.services.chat.get_chat_model", return_value=mock_model):
        with SessionLocal() as db:
            events = list(chat.stream_chat_message(db, user_id=1, course_slug=None, chapter_id=None,
                                                     message="any study tips for today?"))
    assert events == [{"type": "error", "message": "Failed to generate a reply. Please try again."}]
