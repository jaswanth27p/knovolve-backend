"""One-shot custom document generation over course-scoped tools."""
import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy.orm import Session

from app.llm.factory import get_chat_model
from app.llm.prompts import CUSTOM_EXPORT_GENERATE_PROMPT
from app.llm.retry import call_with_retry
from app.models.course import Course
from app.services.export_tools._errors import ExportToolError
from app.services.export_tools.registry import build_export_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8


def _text_content(resp: AIMessage) -> str:
    if isinstance(resp.content, str):
        return resp.content.strip()
    parts = []
    for block in resp.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def generate_custom_markdown(db: Session, course: Course, user_id: int, brief: str, plan: dict) -> str:
    tools = build_export_tools(db, user_id, course.topic_slug)
    tools_by_name = {tool.name: tool for tool in tools}
    model = get_chat_model("custom_export_generate").bind_tools(tools)
    messages: list[BaseMessage] = [
        *CUSTOM_EXPORT_GENERATE_PROMPT.format_messages(
            title=plan["title"],
            plan_json=json.dumps(plan),
            brief=brief,
        )
    ]
    for _ in range(MAX_TOOL_ROUNDS):
        resp = call_with_retry(model.invoke, messages)
        assert isinstance(resp, AIMessage)
        if not resp.tool_calls:
            text = _text_content(resp)
            if text:
                return text
            raise ValueError("custom generation returned an empty document")
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
    raise ValueError("custom generation exceeded tool round budget")
