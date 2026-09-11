import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from app.agents.custom_export import clarify as clarify_agent
from app.agents.custom_export import generate as generate_agent
from app.db import SessionLocal
from app.models.course import Course
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def _seed_custom_course():
    with SessionLocal() as db:
        db.add(Course(topic_slug="custom-agent-course", topic_raw="Custom Agent",
                      topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc)))
        db.commit()


def _course():
    with SessionLocal() as db:
        course = db.query(Course).first()
        assert course is not None
        return course


def _make_model(*invoke_results):
    model = MagicMock()
    model.bind_tools.return_value.invoke = MagicMock(side_effect=list(invoke_results))
    return model


def _fake_tool():
    return SimpleNamespace(name="list_modules", invoke=MagicMock(return_value=[{"id": 1}]))


def test_clarify_uses_tools_and_returns_plan():
    _course()
    model = _make_model(
        AIMessage(content="", tool_calls=[{
            "name": "list_modules", "args": {}, "id": "call_1", "type": "tool_call",
        }]),
        AIMessage(content=json.dumps({
            "type": "plan",
            "reply": "I’ll generate the requested questions.",
            "questions": [],
            "plan": {
                "title": "Interview preparation",
                "output_kind": "qa",
                "length": "medium",
                "item_count": 3,
                "notes": "Questions first, then answers.",
            },
        })),
    )
    with patch("app.agents.custom_export.clarify.get_chat_model", return_value=model), \
        patch("app.agents.custom_export.clarify.build_export_tools", return_value=[_fake_tool()]):
        with SessionLocal() as db:
            result = clarify_agent.clarify_export(db, _course(), 101, "Interview questions", [])
    assert result["type"] == "plan"
    assert result["plan"]["item_count"] == 3


def test_clarify_reasks_once_after_invalid_json():
    _course()
    model = _make_model(
        AIMessage(content="not json"),
        AIMessage(content=json.dumps({
            "type": "clarifying",
            "reply": "How long should this be?",
            "questions": ["How long should this be?"],
            "plan": None,
        })),
    )
    with patch("app.agents.custom_export.clarify.get_chat_model", return_value=model), \
        patch("app.agents.custom_export.clarify.build_export_tools", return_value=[]):
        with SessionLocal() as db:
            result = clarify_agent.clarify_export(db, _course(), 101, "Summarize this", [])
    assert result["type"] == "clarifying"
    assert result["questions"] == ["How long should this be?"]
    assert model.bind_tools.return_value.invoke.call_count == 2


def test_clarify_reasks_once_after_invalid_output_kind():
    _course()
    model = _make_model(
        AIMessage(content=json.dumps({
            "type": "plan",
            "reply": "Here is a plan.",
            "questions": [],
            "plan": {
                "title": "Interview preparation",
                "output_kind": "essay",
                "length": "medium",
                "item_count": 3,
                "notes": None,
            },
        })),
        AIMessage(content=json.dumps({
            "type": "plan",
            "reply": "Here is a plan.",
            "questions": [],
            "plan": {
                "title": "Interview preparation",
                "output_kind": "qa",
                "length": "medium",
                "item_count": 3,
                "notes": None,
            },
        })),
    )
    with patch("app.agents.custom_export.clarify.get_chat_model", return_value=model), \
        patch("app.agents.custom_export.clarify.build_export_tools", return_value=[]):
        with SessionLocal() as db:
            result = clarify_agent.clarify_export(db, _course(), 101, "Interview questions", [])
    assert result["type"] == "plan"
    assert result["plan"]["output_kind"] == "qa"
    assert model.bind_tools.return_value.invoke.call_count == 2


def test_clarify_accepts_json_wrapped_in_prose_and_fences():
    _course()
    fenced = (
        "Sure, here is the plan:\n```json\n"
        + json.dumps({
            "type": "plan",
            "reply": "Ready.",
            "questions": [],
            "plan": {
                "title": "Fenced plan",
                "output_kind": "summary",
                "length": "short",
                "item_count": None,
                "notes": None,
            },
        })
        + "\n```\nLet me know if this works!"
    )
    model = _make_model(AIMessage(content=fenced))
    with patch("app.agents.custom_export.clarify.get_chat_model", return_value=model), \
        patch("app.agents.custom_export.clarify.build_export_tools", return_value=[]):
        with SessionLocal() as db:
            result = clarify_agent.clarify_export(db, _course(), 101, "Summarize this", [])
    assert result["type"] == "plan"
    assert result["plan"]["title"] == "Fenced plan"
    assert model.bind_tools.return_value.invoke.call_count == 1


def test_clarify_budget_hit_raises():
    _course()
    model = _make_model(*[AIMessage(content="", tool_calls=[{
        "name": "missing_tool", "args": {}, "id": "call_x", "type": "tool_call",
    }])] * clarify_agent.MAX_TOOL_ROUNDS)
    with patch("app.agents.custom_export.clarify.get_chat_model", return_value=model), \
        patch("app.agents.custom_export.clarify.build_export_tools", return_value=[]):
        with SessionLocal() as db:
            try:
                clarify_agent.clarify_export(db, _course(), 101, "spam", [])
                raise AssertionError("expected ValueError")
            except ValueError:
                pass


def test_generate_returns_markdown_after_tool_lookup():
    _course()
    model = _make_model(
        AIMessage(content="", tool_calls=[{
            "name": "list_modules", "args": {}, "id": "call_1", "type": "tool_call",
        }]),
        AIMessage(content="# Summary\n\nCourse summary."),
    )
    plan = {
        "title": "Short summary",
        "output_kind": "summary",
        "length": "short",
        "item_count": None,
        "notes": None,
    }
    with patch("app.agents.custom_export.generate.get_chat_model", return_value=model), \
        patch("app.agents.custom_export.generate.build_export_tools", return_value=[_fake_tool()]):
        with SessionLocal() as db:
            markdown = generate_agent.generate_custom_markdown(
                db, _course(), 101, "Short summary version of this course.", plan
            )
    assert markdown.startswith("# Summary")
