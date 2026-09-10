import pytest
from pydantic import ValidationError

from app.schemas.chat import (
    MAX_HISTORY_TURNS, MAX_MESSAGE_CHARS, ChatRequest, ChatTurn, RouteContext,
)


def test_chat_request_defaults_to_empty_history_and_no_context():
    req = ChatRequest(message="hi")
    assert req.history == []
    assert req.context is None
    assert req.current_route is None


def test_chat_request_accepts_history_and_route():
    req = ChatRequest(
        message="what's my progress",
        history=[ChatTurn(role="user", content="hey"), ChatTurn(role="assistant", content="hi!")],
        current_route=RouteContext(page="chapter_content", course_slug="python-basics", chapter_id=3),
    )
    assert req.history[0].role == "user"
    assert req.current_route is not None
    assert req.current_route.chapter_id == 3


def test_chat_request_rejects_oversized_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * (MAX_MESSAGE_CHARS + 1))


def test_chat_request_rejects_too_many_history_turns():
    turns = [ChatTurn(role="user", content="hi")] * (MAX_HISTORY_TURNS + 1)
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", history=turns)
