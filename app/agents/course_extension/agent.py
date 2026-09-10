"""Plans new extension chapters for a course.

Bounded `bind_tools` loop over the read-only chapter-content tool (the same
idiom as `app/services/chat.py`), finishing with a JSON structured output that
is reparsed once on a schema violation. Runs inside the Celery job so
transient LLM failures are retried via `call_with_retry` per invoke.
"""
import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ValidationError
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.llm.factory import get_chat_model
from app.llm.prompts import EXTENSION_PLAN_PROMPT
from app.llm.retry import call_with_retry
from app.models.course import Chapter, Course, Module
from app.services.chat_tools import chapters as chat_chapters

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5
MAX_PARSE_ATTEMPTS = 2
MAX_CHAPTERS = 8


class ExtensionChapterDraft(BaseModel):
    title: str
    objective: str


class ExtensionPlanResponse(BaseModel):
    chapters: list[ExtensionChapterDraft]


def _outline_json(db: Session, course: Course, user_id: int) -> str:
    """Full outline context: global modules -> chapters with ids, then the
    caller's own bucket chapters. Never includes another user's bucket."""
    modules = db.query(Module).filter(
        Module.course_id == course.id,
        or_(
            Module.scope == "global",
            and_(Module.scope == "user", Module.user_id == user_id),
        ),
    ).order_by(Module.order).all()
    out = []
    for m in modules:
        chapters = db.query(Chapter).filter_by(module_id=m.id).order_by(Chapter.order).all()
        if m.scope == "user" and not chapters:
            continue
        out.append({
            "id": m.id, "title": m.title, "objective": m.objective,
            "is_additional": m.scope == "user",
            "chapters": [{"id": c.id, "title": c.title, "objective": c.objective} for c in chapters],
        })
    return json.dumps(out)


def _build_tool(db: Session, course: Course, user_id: int):
    @tool
    def read_chapter_content(chapter_id: int) -> dict:
        """Read an existing chapter's generated content (if any) to judge
        whether a topic is already covered. Never triggers generation."""
        return chat_chapters.get_chapter_content(db, user_id, course.topic_slug, chapter_id)

    return read_chapter_content


def _parse_chapters(resp: AIMessage, model, messages: list[BaseMessage]) -> list[dict]:
    for _ in range(MAX_PARSE_ATTEMPTS):
        content = resp.content
        if isinstance(content, str):
            try:
                parsed = ExtensionPlanResponse.model_validate_json(content)
                return [c.model_dump() for c in parsed.chapters[:MAX_CHAPTERS]]
            except (ValidationError, ValueError):
                logger.warning("extension plan returned unparseable output; re-asking")
        messages.append(resp)
        messages.append(HumanMessage(
            content="Your previous reply was not valid JSON matching the required schema. "
                    "Reply with the JSON object only."
        ))
        resp = call_with_retry(model.invoke, messages)
        assert isinstance(resp, AIMessage)
    raise ValueError("extension plan produced invalid JSON after corrective retries")


def plan_new_chapters(db: Session, course: Course, user_id: int, request_text: str) -> list[dict]:
    read_tool = _build_tool(db, course, user_id)
    model = get_chat_model("extension_plan").bind_tools([read_tool])
    messages: list[BaseMessage] = [
        *EXTENSION_PLAN_PROMPT.format_messages(
            outline_json=_outline_json(db, course, user_id), request=request_text,
        ),
    ]
    for _ in range(MAX_TOOL_ROUNDS):
        resp = call_with_retry(model.invoke, messages)
        assert isinstance(resp, AIMessage)
        if not resp.tool_calls:
            return _parse_chapters(resp, model, messages)
        messages.append(resp)
        for call in resp.tool_calls:
            if call["name"] == "read_chapter_content":
                try:
                    result = read_tool.invoke(call["args"])
                except Exception as exc:  # noqa: BLE001 - model-facing recovery
                    logger.warning("read_chapter_content failed: %s", exc)
                    result = "Error: something went wrong calling this tool."
            else:
                result = f"Error: no such tool '{call['name']}'."
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    raise ValueError("extension plan exceeded tool round budget")
