"""Floating chatbot: a tool-calling agent. The client builds/caches a coarse
LearnerContextBundle on the first turn of a conversation (nothing is
persisted server-side, see docs/superpowers/specs/2026-09-10-agentic-chat-
context-design.md) and resends it on every later turn; the agent seeds its
system prompt from that bundle and calls into app.services.chat_tools for
anything deeper or more current, always live against the DB."""
import logging
from typing import Iterator

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from sqlalchemy.orm import Session

from app.llm.factory import get_chat_model
from app.llm.langfuse_client import traced_workflow
from app.llm.prompts import CHAT_AGENT_SYSTEM_PROMPT
from app.llm.retry import call_with_retry
from app.schemas.chat import ChatRequest, ChatResponse, ChatTurn, LearnerContextBundle
from app.services.chat_context import build_context_bundle
from app.services.chat_tools._errors import ChatToolError
from app.services.chat_tools.registry import build_tools

logger = logging.getLogger(__name__)

# Hard cap on tool-call round-trips per message so a model that keeps calling
# tools (bad args, a tool that never satisfies it) can't loop forever.
MAX_TOOL_ROUNDS = 5

# Only the most recent turns actually reach the model. The request schema
# accepts a longer history (MAX_HISTORY_TURNS) so the client can hold the full
# conversation, but the prompt is deliberately bounded so a long-running
# conversation can't grow the prompt without limit.
MAX_PROMPT_HISTORY_TURNS = 20

# Used whenever the model never produces any text of its own — it burned through
# MAX_TOOL_ROUNDS still asking for tools, or streamed zero tokens. Better than
# handing the user a blank reply with no signal that anything went wrong.
FALLBACK_REPLY = "I wasn't able to finish looking that up — try rephrasing your question?"

#: What `ChatOpenAI.bind_tools(...)` hands back.
BoundModel = Runnable[LanguageModelInput, BaseMessage]


def _history_to_messages(history: list[ChatTurn]) -> list[BaseMessage]:
    recent = history[-MAX_PROMPT_HISTORY_TURNS:]
    return [
        HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content)
        for turn in recent
    ]


def _execute_tool_call(tools_by_name: dict[str, BaseTool], call: ToolCall) -> ToolMessage:
    """Turn one model tool call into a model-facing ToolMessage. Every failure
    is caught and returned as text so a bad/hallucinated call can't 500 the
    request (or truncate an already-started stream)."""
    tool_fn = tools_by_name.get(call["name"])
    if tool_fn is None:
        # The model asked for a tool that isn't bound (hallucinated or
        # renamed). Tell it so it can recover instead of 500-ing.
        result = f"Error: no such tool '{call['name']}'."
    else:
        try:
            result = tool_fn.invoke(call["args"])
        except ChatToolError as exc:
            # ChatToolError messages are written to be model-facing.
            result = f"Error: {exc}"
        except Exception:  # noqa: BLE001 - bad/hallucinated args (pydantic
            # ValidationError) or any runtime/DB failure inside a tool must
            # not 500 the request; the model gets a chance to recover.
            logger.warning("chat tool %r failed", call["name"], exc_info=True)
            result = "Error: something went wrong calling this tool."
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def _run_tool_rounds(
    tools_by_name: dict[str, BaseTool], model: BoundModel, messages: list[BaseMessage],
) -> AIMessage | None:
    """Mutates `messages` in place, executing tool-call rounds until the model
    responds with no tool_calls, and returns that final response (which is
    deliberately NOT appended to `messages`). Returns None if MAX_TOOL_ROUNDS
    was hit while the model was still asking for tools. Used only by the
    non-streaming `answer_chat_message`."""
    for _ in range(MAX_TOOL_ROUNDS):
        resp = call_with_retry(model.invoke, messages)
        assert isinstance(resp, AIMessage)
        if not resp.tool_calls:
            return resp
        messages.append(resp)
        for call in resp.tool_calls:
            messages.append(_execute_tool_call(tools_by_name, call))
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
    with traced_workflow("Chat Reply", user_id=user_id, tags=["chat"]):
        bundle = req.context if req.context is not None else build_context_bundle(db, user_id, req.current_route)
        model, messages, final = _build_agent(db, user_id, req, bundle)
        # `final` is already the model's tool-free answer; only re-invoke when the
        # round cap cut the loop short and we never got one.
        resp: BaseMessage = final if final is not None else call_with_retry(model.invoke, messages)
        content = resp.content
        if not isinstance(content, str):
            raise TypeError(f"expected str content from LLM response, got {type(content)}")
        # The fallback re-invoke can itself come back with tool_calls and empty text.
        reply = content.strip() or FALLBACK_REPLY
        return ChatResponse(reply=reply, context=None if req.context is not None else bundle)


def stream_chat_message(db: Session, user_id: int, req: ChatRequest) -> Iterator[dict]:
    # Streaming-first: the model's first response is streamed straight to the
    # client. Only if that response actually asks for tools do we run a round
    # and stream again. The old code buffered a full non-streaming answer here
    # (`_build_agent` -> `_run_tool_rounds`) and then discarded it to re-run
    # `model.stream`, generating every simple reply twice (~2x latency).
    #
    # Everything that can fail lives inside the try: the response has already
    # started streaming once we yield, so any failure from here on has to reach
    # the client as an `error` event, not escape the generator and truncate the
    # stream (this includes building the bundle on the first turn).
    with traced_workflow("Chat Reply", user_id=user_id, tags=["chat", "streaming"]):
        yield from _stream_chat_message(db, user_id, req)


def _stream_chat_message(db: Session, user_id: int, req: ChatRequest) -> Iterator[dict]:
    try:
        bundle = req.context if req.context is not None else build_context_bundle(db, user_id, req.current_route)
        if req.context is None:
            yield {"type": "context", "bundle": bundle.model_dump(mode="json")}

        tools = build_tools(db, user_id)
        tools_by_name = {t.name: t for t in tools}
        base_model = get_chat_model("chat_reply")
        model = base_model.bind_tools(tools)
        route_json = req.current_route.model_dump_json() if req.current_route is not None else "unknown"
        system = CHAT_AGENT_SYSTEM_PROMPT.format_messages(
            bundle_json=bundle.model_dump_json(), route_json=route_json,
        )
        messages: list[BaseMessage] = [
            *system, *_history_to_messages(req.history), HumanMessage(content=req.message),
        ]

        yielded_any = False
        for _ in range(MAX_TOOL_ROUNDS):
            aggregated: AIMessageChunk | None = None
            for chunk in model.stream(messages):
                if isinstance(chunk, AIMessageChunk):
                    aggregated = chunk if aggregated is None else aggregated + chunk
                content = chunk.content
                if isinstance(content, str) and content:
                    yielded_any = True
                    yield {"type": "token", "text": content}
            if aggregated is None or not aggregated.tool_calls:
                break
            messages.append(aggregated)
            for call in aggregated.tool_calls:
                messages.append(_execute_tool_call(tools_by_name, call))

        if not yielded_any:
            # The model streamed no text at all (it may have gone straight to
            # tools every round, or come back empty). Force one tool-free
            # answer rather than leaving the learner with a blank reply.
            resp = call_with_retry(base_model.invoke, messages)
            content = resp.content if isinstance(resp.content, str) else ""
            yield {"type": "token", "text": content.strip() or FALLBACK_REPLY}
    except Exception:  # noqa: BLE001 - surfaced to the client as a chat error, not a 500
        yield {"type": "error", "message": "Failed to generate a reply. Please try again."}
        return
    yield {"type": "done"}
