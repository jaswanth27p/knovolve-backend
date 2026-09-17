import uuid
from typing import cast
from pydantic import SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings
from app.llm.providers import PROVIDERS
from app.llm.config import LLM_NODES, EMBEDDING_NODE
from app.llm.langfuse_client import get_langfuse_handler
from app.llm.retry import call_with_retry

# OpenCode Go rejects requests with no identifying client headers
# ("MissingSessionID") — it's designed for interactive coding-agent CLIs,
# which send their own User-Agent and a stable per-conversation
# `x-opencode-session` id (see https://opencode.ai/docs/go/#where-can-i-use-it).
# This backend isn't a coding-agent session, but one process-lifetime id is a
# reasonable stand-in for "stable session": every LLM call from this running
# server shares it, satisfying the requirement without threading a per-request
# session id through every node. Harmless to send to the other providers too.
_PROCESS_SESSION_ID = str(uuid.uuid4())
_DEFAULT_HEADERS = {
    "User-Agent": "knovolve-backend/1.0",
    "x-opencode-session": _PROCESS_SESSION_ID,
}

def get_chat_model(node: str) -> ChatOpenAI:
    cfg = LLM_NODES[node]
    provider = PROVIDERS[cfg["provider"]]
    model = ChatOpenAI(
        base_url=provider["base_url"],
        api_key=SecretStr(getattr(settings, provider["settings_field"])),
        model=cfg["model"],
        request_timeout=settings.llm_request_timeout_seconds,  # pyright: ignore[reportCallIssue]
        default_headers=_DEFAULT_HEADERS,
    )
    handler = get_langfuse_handler()
    if handler is not None:
        # .with_config(...) statically returns a generic
        # Runnable[LanguageModelInput, AIMessage] per langchain-core's stubs,
        # but at runtime it's a RunnableBinding whose __getattr__ delegates
        # every attribute/method access — bind_tools, with_structured_output,
        # openai_api_base, .config, everything — straight through to the
        # wrapped ChatOpenAI instance (verified empirically). Declaring the
        # return type as a ChatOpenAI | Runnable union instead of casting
        # here would be "more honest" in isolation, but it erases
        # ChatOpenAI-specific members from every one of this function's 41
        # call sites project-wide (bind_tools/with_structured_output aren't
        # defined on the generic Runnable base), which is a real regression,
        # not just noise — confirmed via a full `pyright` run. This cast
        # documents a structurally-verified runtime fact instead of
        # silencing a real error, and keeps every existing call site's
        # typing unchanged.
        model = cast(
            ChatOpenAI,
            model.with_config({"callbacks": [handler], "metadata": {"langfuse_tags": [node]}}),
        )
    return model

def _embeddings_client() -> OpenAIEmbeddings:
    provider = PROVIDERS[EMBEDDING_NODE["provider"]]
    client = OpenAIEmbeddings(
        base_url=provider["base_url"],
        api_key=SecretStr(getattr(settings, provider["settings_field"])),
        model=EMBEDDING_NODE["model"],
        request_timeout=settings.llm_request_timeout_seconds,  # pyright: ignore[reportCallIssue]
        default_headers=_DEFAULT_HEADERS,
        # This provider's embedding endpoint only accepts raw text, not the
        # tiktoken-encoded integer token arrays OpenAIEmbeddings sends by
        # default ("Invalid input format. Nvidia embeddings support strings
        # and multimodal inputs") — bypass tokenization and send text as-is.
        check_embedding_ctx_length=False,
        # ...and it doesn't support the base64 encoding_format OpenAIEmbeddings
        # also defaults to ("Nvidia embeddings do not support base64
        # encoding_format") — request plain floats instead.
        encoding_format="float",
    )
    # Unlike ChatOpenAI, OpenAIEmbeddings (langchain-openai 1.6.0) does NOT
    # inherit from Runnable and has no with_config()/callbacks support at all
    # (no `callbacks`/`metadata` pydantic fields, no callback/run_manager
    # plumbing anywhere in its source — confirmed by inspection). Calling
    # `.with_config(...)` on it raises AttributeError at runtime, so there is
    # currently no way to bind a Langfuse callback handler onto an embeddings
    # client the way get_chat_model does for ChatOpenAI above. Deliberately
    # not attempted here — see task-6-report.md for the runtime proof and
    # follow-up options (e.g. a Langfuse `@observe`-decorated wrapper around
    # embed(), tracked separately from this factory-level binding).
    return client

def embed(text: str) -> list[float]:
    client = _embeddings_client()
    return call_with_retry(client.embed_query, text)
