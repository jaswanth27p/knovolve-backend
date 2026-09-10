from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.db import SessionLocal
from app.schemas.chat import ChatRequest, ChatTurn
from app.services import chat


def _bound_model(responses: list[AIMessage]):
    """A MagicMock standing in for `ChatOpenAI.bind_tools(...)`: .invoke()
    returns the next response in `responses` each call; .bind_tools()
    returns itself so `get_chat_model(...).bind_tools(tools)` chains through."""
    model = MagicMock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = responses
    return model


def test_answer_returns_bundle_only_when_request_had_none(monkeypatch):
    model = _bound_model([AIMessage(content="You're doing great!")])
    monkeypatch.setattr("app.services.chat.get_chat_model", lambda node: model)
    monkeypatch.setattr("app.services.chat.build_tools", lambda db, user_id: [])
    with SessionLocal() as db:
        response_with_none = chat.answer_chat_message(db, user_id=1, req=ChatRequest(message="hi"))
    assert response_with_none.context is not None

    model2 = _bound_model([AIMessage(content="Still great!")])
    monkeypatch.setattr("app.services.chat.get_chat_model", lambda node: model2)
    with SessionLocal() as db:
        response_with_context = chat.answer_chat_message(
            db, user_id=1,
            req=ChatRequest(message="hi again", context=response_with_none.context),
        )
    assert response_with_context.context is None


def test_answer_executes_tool_call_then_returns_final_reply(monkeypatch):
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "get_user_stats", "args": {}, "id": "call-1", "type": "tool_call"}],
    )
    final_response = AIMessage(content="You've done 1 assignment today.")
    model = _bound_model([tool_call_response, final_response])

    fake_tool = MagicMock()
    fake_tool.name = "get_user_stats"
    fake_tool.invoke.return_value = {"assignments_attempted_today": 1}

    monkeypatch.setattr("app.services.chat.get_chat_model", lambda node: model)
    monkeypatch.setattr("app.services.chat.build_tools", lambda db, user_id: [fake_tool])

    with SessionLocal() as db:
        response = chat.answer_chat_message(db, user_id=1, req=ChatRequest(message="how many assignments today?"))

    fake_tool.invoke.assert_called_once_with({})
    assert response.reply == "You've done 1 assignment today."


def test_answer_turns_tool_error_into_tool_message_not_a_crash(monkeypatch):
    from app.services.chat_tools._errors import ChatToolError

    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "get_chapter_progress", "args": {"course_slug": "x", "chapter_id": 1},
                      "id": "call-1", "type": "tool_call"}],
    )
    final_response = AIMessage(content="You haven't started that course yet.")
    model = _bound_model([tool_call_response, final_response])

    failing_tool = MagicMock()
    failing_tool.name = "get_chapter_progress"
    failing_tool.invoke.side_effect = ChatToolError("You haven't started the course 'X' yet.")

    monkeypatch.setattr("app.services.chat.get_chat_model", lambda node: model)
    monkeypatch.setattr("app.services.chat.build_tools", lambda db, user_id: [failing_tool])

    with SessionLocal() as db:
        response = chat.answer_chat_message(
            db, user_id=1, req=ChatRequest(message="am I done with chapter 1?"),
        )
    assert response.reply == "You haven't started that course yet."


def test_stream_yields_context_event_on_first_turn_only(monkeypatch):
    final = AIMessage(content="ok")
    model = _bound_model([final])
    model.stream.return_value = [MagicMock(content="ok")]
    monkeypatch.setattr("app.services.chat.get_chat_model", lambda node: model)
    monkeypatch.setattr("app.services.chat.build_tools", lambda db, user_id: [])
    with SessionLocal() as db:
        events = list(chat.stream_chat_message(db, user_id=1, req=ChatRequest(message="hi")))
    assert events[0]["type"] == "context"
    assert "in_progress_count" in events[0]["bundle"]
    assert events[-1] == {"type": "done"}


def test_stream_omits_context_event_when_request_already_has_one(monkeypatch):
    final = AIMessage(content="ok again")
    model = _bound_model([final])
    model.stream.return_value = [MagicMock(content="ok again")]
    monkeypatch.setattr("app.services.chat.get_chat_model", lambda node: model)
    monkeypatch.setattr("app.services.chat.build_tools", lambda db, user_id: [])

    with SessionLocal() as db:
        bundle = chat.answer_chat_message(db, user_id=1, req=ChatRequest(message="hi")).context
    assert bundle is not None

    model2 = _bound_model([AIMessage(content="ok again")])
    model2.stream.return_value = [MagicMock(content="ok again")]
    monkeypatch.setattr("app.services.chat.get_chat_model", lambda node: model2)
    with SessionLocal() as db:
        events = list(chat.stream_chat_message(
            db, user_id=1,
            req=ChatRequest(message="hi again", context=bundle, history=[ChatTurn(role="user", content="hi")]),
        ))
    assert events[0]["type"] == "token"
    assert events[-1] == {"type": "done"}
