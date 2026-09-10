import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.agents.course_extension.agent import MAX_TOOL_ROUNDS, plan_new_chapters
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.user import User


@pytest.fixture(autouse=True)
def _seed_user():
    # test_outline_includes_global_and_user_bucket writes user-scoped module
    # rows via append_chapters, which FKs user_id -> users.id (see
    # tests/test_course_extension_service.py for the same seed).
    with SessionLocal() as db:
        db.add(User(id=5, email="ext-agent-user-5@example.com", password_hash="x"))
        db.commit()


def _make_course():
    with SessionLocal() as db:
        course = Course(topic_slug="ext-agent-course", topic_raw="Ext Agent",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.flush()
        m = Module(course_id=course.id, title="M1", objective="o", order=1, scope="global")
        db.add(m)
        db.flush()
        db.add(Chapter(module_id=m.id, title="Intro", objective="o", order=1, scope="global"))
        db.commit()
        db.refresh(course)
        return course


def _make_model(*invoke_results):
    """Return a MagicMock whose bound-tools sub-mock drives all invokes.

    The agent calls `get_chat_model(...).bind_tools([...])` and invokes the
    BOUND object, so the mock must be wired on `model.bind_tools.return_value`
    (the plain `model.invoke` is never called)."""
    model = MagicMock()
    model.bind_tools.return_value.invoke = MagicMock(side_effect=list(invoke_results))
    return model


def test_uses_tool_loop_and_returns_structured_chapters():
    course = _make_course()
    model = _make_model(
        AIMessage(content="", tool_calls=[{
            "name": "read_chapter_content", "args": {"chapter_id": 1}, "id": "call_1", "type": "tool_call",
        }]),
        AIMessage(content=json.dumps({"chapters": [
            {"title": "TCP Handshake", "objective": "Explain the three-way handshake."},
        ]})),
    )
    with patch("app.agents.course_extension.agent.get_chat_model", return_value=model) as mock_get, \
         patch("app.services.chat_tools.chapters.get_chapter_content",
               return_value={"available": False, "reason": "not generated"}) as mock_tool:
        with SessionLocal() as db:
            result = plan_new_chapters(db, course, 5, "teach me tcp")
    mock_get.assert_called_once_with("extension_plan")
    mock_tool.assert_called_once()
    assert result == [{"title": "TCP Handshake", "objective": "Explain the three-way handshake."}]


def test_direct_json_without_tools():
    course = _make_course()
    model = _make_model(AIMessage(content=json.dumps({"chapters": [{"title": "X", "objective": "o"}]})))
    with patch("app.agents.course_extension.agent.get_chat_model", return_value=model):
        with SessionLocal() as db:
            result = plan_new_chapters(db, course, 5, "add x")
    assert result == [{"title": "X", "objective": "o"}]


def test_invalid_json_triggers_one_corrective_reinvoke():
    course = _make_course()
    model = _make_model(
        AIMessage(content="not json at all"),
        AIMessage(content=json.dumps({"chapters": [{"title": "Y", "objective": "o"}]})),
    )
    with patch("app.agents.course_extension.agent.get_chat_model", return_value=model):
        with SessionLocal() as db:
            result = plan_new_chapters(db, course, 5, "add y")
    assert result == [{"title": "Y", "objective": "o"}]
    assert model.bind_tools.return_value.invoke.call_count == 2


def test_tool_loop_budget_hit_raises():
    course = _make_course()
    # The model never stops asking for tools, so it is invoked once per round
    # until the budget is exhausted; provide one response per round.
    model = _make_model(*[AIMessage(content="", tool_calls=[{
        "name": "read_chapter_content", "args": {}, "id": "call_x", "type": "tool_call",
    }])] * MAX_TOOL_ROUNDS)
    with patch("app.agents.course_extension.agent.get_chat_model", return_value=model), \
         patch("app.services.chat_tools.chapters.get_chapter_content",
               return_value={"available": False}):
        with SessionLocal() as db:
            try:
                plan_new_chapters(db, course, 5, "spam")
                raise AssertionError("expected ValueError")
            except ValueError:
                pass


def test_outline_includes_global_and_user_bucket():
    course_with_bucket = _make_course()
    with SessionLocal() as db:
        from app.models.course_extension import CourseExtensionJob  # noqa: F401
        from app.services.course_extension import append_chapters
        append_chapters(db, 5, course_with_bucket, [{"title": "My Ext", "objective": "o"}])
        db.commit()
    model = _make_model(AIMessage(content=json.dumps({"chapters": []})))
    with patch("app.agents.course_extension.agent.get_chat_model", return_value=model):
        with SessionLocal() as db:
            plan_new_chapters(db, course_with_bucket, 5, "nothing")
    # The prompt payload carried both the global chapter and the bucket
    prompt_msg = model.bind_tools.return_value.invoke.call_args.args[0][-1]
    assert "Intro" in prompt_msg.content
    assert "My Ext" in prompt_msg.content
