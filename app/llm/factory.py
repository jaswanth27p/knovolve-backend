import uuid
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
        # NOT model.with_config({"callbacks": [handler], ...}) — that wraps
        # `model` in a RunnableBinding, and RunnableBinding.__getattr__ only
        # merges the wrapper's .config into delegated methods whose signature
        # has a `config` parameter (see langchain_core.runnables.base:
        # RunnableBinding.__getattr__). Neither bind_tools() nor
        # with_structured_output() takes `config` — both are plain
        # `return self.bind(...)` calls forwarded straight to the *inner*
        # ChatOpenAI instance, so `self` inside them is the unwrapped model
        # and the returned RunnableBinding gets a fresh, empty .config,
        # silently dropping the callbacks/metadata set here. Every call site
        # that does `get_chat_model(...).bind_tools(...)` or
        # `.with_structured_output(...)` — chat, custom export, course
        # extension, and effectively every generation node in course
        # creation/chapter content/assignment/evaluation — would produce zero
        # Langfuse spans despite `traced_workflow` being entered correctly.
        # Verified empirically: live-server /me/chat, chapter content
        # generation, and grep of with_structured_output's own source (also
        # `self.bind(**kwargs)`) all reproduced the drop; a direct repro with
        # `model.with_config(...)` then `.bind_tools(...)` produced zero
        # captured spans in a patched LangfuseSpanProcessor exporter.
        #
        # callbacks/metadata/tags are native Pydantic fields on
        # BaseChatModel (see langchain_core.language_models.chat_models),
        # not RunnableBinding-only config. Setting them directly on the
        # ChatOpenAI instance means `self` inside bind_tools()/
        # with_structured_output() IS this already-configured instance, so
        # every RunnableBinding they construct wraps a model that already
        # carries the handler — confirmed empirically to survive both.
        model.callbacks = [handler]
        model.metadata = {"langfuse_tags": [node]}
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
