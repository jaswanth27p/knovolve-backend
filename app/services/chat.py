"""Floating chatbot: a tool-calling agent. The client builds/caches a coarse
LearnerContextBundle on the first turn of a conversation (nothing is
persisted server-side, see docs/superpowers/specs/2026-09-10-agentic-chat-
context-design.md) and resends it on every later turn; the agent seeds its
system prompt from that bundle and calls into app.services.chat_tools for
anything deeper or more current, always live against the DB."""
from typing import Iterator

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from sqlalchemy.orm import Session

from app.llm.factory import get_chat_model
from app.llm.prompts import CHAT_AGENT_SYSTEM_PROMPT
from app.llm.retry import call_with_retry
from app.schemas.chat import ChatRequest, ChatResponse, ChatTurn, LearnerContextBundle
from app.services.chat_context import build_context_bundle
from app.services.chat_tools._errors import ChatToolError
from app.services.chat_tools.registry import build_tools

# Hard cap on tool-call round-trips per message so a model that keeps calling
# tools (bad args, a tool that never satisfies it) can't loop forever.
MAX_TOOL_ROUNDS = 5

#: What `ChatOpenAI.bind_tools(...)` hands back.
BoundModel = Runnable[LanguageModelInput, BaseMessage]


def _history_to_messages(history: list[ChatTurn]) -> list[BaseMessage]:
    return [
        HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content)
        for turn in history
    ]


def _run_tool_rounds(
    tools_by_name: dict[str, BaseTool], model: BoundModel, messages: list[BaseMessage],
) -> AIMessage | None:
    """Mutates `messages` in place, executing tool-call rounds until the model
    responds with no tool_calls, and returns that final response (which is
    deliberately NOT appended to `messages`). Returns None if MAX_TOOL_ROUNDS
    was hit while the model was still asking for tools.

    The buffered final response is what the non-streaming caller replies with;
    the streaming caller throws it away and re-runs the same `messages`
    through `.stream()` so the reply arrives token-by-token."""
    for _ in range(MAX_TOOL_ROUNDS):
        resp = call_with_retry(model.invoke, messages)
        assert isinstance(resp, AIMessage)
        if not resp.tool_calls:
            return resp
        messages.append(resp)
        for call in resp.tool_calls:
            tool_fn = tools_by_name.get(call["name"])
            if tool_fn is None:
                # The model asked for a tool that isn't bound (hallucinated or
                # renamed). Tell it so it can recover instead of 500-ing.
                result = f"Error: no such tool '{call['name']}'."
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except ChatToolError as exc:
                    result = f"Error: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return None


def _build_agent(
    db: Session, user_id: int, req: ChatRequest, bundle: LearnerContextBundle,
) -> tuple[BoundModel, list[BaseMessage], AIMessage | None]:
    tools = build_tools(db, user_id)
    tools_by_name = {t.name: t for t in tools}
    model = get_chat_model("chat_reply").bind_tools(tools)
    route_json = req.current_route.model_dump_json() if req.current_route is not None else "unknown"
    system = CHAT_AGENT_SYSTEM_PROMPT.format_messages(
        bundle_json=bundle.model_dump_json(), route_json=route_json,
    )
    messages: list[BaseMessage] = [*system, *_history_to_messages(req.history), HumanMessage(content=req.message)]
    final = _run_tool_rounds(tools_by_name, model, messages)
    return model, messages, final


def answer_chat_message(db: Session, user_id: int, req: ChatRequest) -> ChatResponse:
    bundle = req.context if req.context is not None else build_context_bundle(db, user_id, req.current_route)
    model, messages, final = _build_agent(db, user_id, req, bundle)
    # `final` is already the model's tool-free answer; only re-invoke when the
    # round cap cut the loop short and we never got one.
    resp: BaseMessage = final if final is not None else call_with_retry(model.invoke, messages)
    content = resp.content
    if not isinstance(content, str):
        raise TypeError(f"expected str content from LLM response, got {type(content)}")
    return ChatResponse(reply=content.strip(), context=None if req.context is not None else bundle)


def stream_chat_message(db: Session, user_id: int, req: ChatRequest) -> Iterator[dict]:
    bundle = req.context if req.context is not None else build_context_bundle(db, user_id, req.current_route)
    if req.context is None:
        yield {"type": "context", "bundle": bundle.model_dump(mode="json")}

    model, messages, _ = _build_agent(db, user_id, req, bundle)
    try:
        for chunk in model.stream(messages):
            content = chunk.content
            if isinstance(content, str) and content:
                yield {"type": "token", "text": content}
    except Exception:  # noqa: BLE001 - surfaced to the client as a chat error, not a 500
        yield {"type": "error", "message": "Failed to generate a reply. Please try again."}
        return
    yield {"type": "done"}
