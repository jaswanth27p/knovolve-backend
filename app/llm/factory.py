import uuid
from pydantic import SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings
from app.llm.providers import PROVIDERS
from app.llm.config import LLM_NODES, EMBEDDING_NODE
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
    return ChatOpenAI(
        base_url=provider["base_url"],
        api_key=SecretStr(getattr(settings, provider["settings_field"])),
        model=cfg["model"],
        request_timeout=settings.llm_request_timeout_seconds,  # pyright: ignore[reportCallIssue]
        default_headers=_DEFAULT_HEADERS,
    )

def _embeddings_client() -> OpenAIEmbeddings:
    provider = PROVIDERS[EMBEDDING_NODE["provider"]]
    return OpenAIEmbeddings(
        base_url=provider["base_url"],
        api_key=SecretStr(getattr(settings, provider["settings_field"])),
        model=EMBEDDING_NODE["model"],
        request_timeout=settings.llm_request_timeout_seconds,  # pyright: ignore[reportCallIssue]
        default_headers=_DEFAULT_HEADERS,
    )

def embed(text: str) -> list[float]:
    client = _embeddings_client()
    return call_with_retry(client.embed_query, text)
