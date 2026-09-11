"""Stateless custom-export clarification over course-scoped tools."""
import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.llm.factory import get_chat_model
from app.llm.prompts import CUSTOM_EXPORT_CLARIFY_PROMPT
from app.llm.retry import call_with_retry
from app.models.course import Course
from app.schemas.chat import ChatTurn
from app.agents.course_extension.agent import _outline_json as _existing_outline_json
from app.services.export_tools._errors import ExportToolError
from app.services.export_tools.registry import build_export_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
MAX_PARSE_ATTEMPTS = 2
MAX_PROMPT_HISTORY_TURNS = 20


class ExportPlanDraft(BaseModel):
    title: str
    output_kind: Literal["summary", "qa", "cheat_sheet", "custom"]
    length: Literal["short", "medium", "long"]
    item_count: int | None = None
    notes: str | None = None


class ClarifyResultDraft(BaseModel):
    type: Literal["clarifying", "plan"]
    reply: str
    questions: list[str] = []
    plan: ExportPlanDraft | None = None


def _outline_json(db: Session, course: Course, user_id: int) -> str:
    return _existing_outline_json(db, course, user_id)


def _parse_result(resp: AIMessage, model, messages: list[BaseMessage]) -> dict:
    for _ in range(MAX_PARSE_ATTEMPTS):
        if isinstance(resp.content, str):
            try:
                return ClarifyResultDraft.model_validate_json(resp.content).model_dump()
            except (ValidationError, ValueError):
                logger.warning("custom clarification returned invalid JSON; re-asking")
        messages.append(resp)
        messages.append(HumanMessage(content="Reply with JSON only, matching the required schema."))
        resp = call_with_retry(model.invoke, messages)
        assert isinstance(resp, AIMessage)
    raise ValueError("custom clarification produced invalid JSON after corrective retries")


def clarify_export(
    db: Session, course: Course, user_id: int, message: str, history: list[ChatTurn],
) -> dict:
    tools = build_export_tools(db, user_id, course.topic_slug)
    tools_by_name = {tool.name: tool for tool in tools}
    model = get_chat_model("custom_export_clarify").bind_tools(tools)
    history_payload = [turn.model_dump() for turn in history[-MAX_PROMPT_HISTORY_TURNS:]]
    messages: list[BaseMessage] = CUSTOM_EXPORT_CLARIFY_PROMPT.format_messages(
        outline_json=_outline_json(db, course, user_id),
        history_json=json.dumps(history_payload),
        message=message,
    )
    for _ in range(MAX_TOOL_ROUNDS):
        resp = call_with_retry(model.invoke, messages)
        assert isinstance(resp, AIMessage)
        if not resp.tool_calls:
            return _parse_result(resp, model, messages)
        messages.append(resp)
        for call in resp.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = f"Error: no such tool '{call['name']}'."
            else:
                try:
                    result = json.dumps(tool.invoke(call["args"]), default=str)
                except ExportToolError as exc:
                    result = f"Error: {exc}"
                except Exception:
                    logger.warning("custom export tool %r failed", call["name"], exc_info=True)
                    result = "Error: something went wrong calling this tool."
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    raise ValueError("custom clarification exceeded tool round budget")
