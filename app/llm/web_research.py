"""Web research tools for grounding LLM generation in live sources.

DuckDuckGo search (via ``ddgs``) discovers candidate URLs; page text is
extracted with ``trafilatura``. No embeddings or vector store: fetched text is
handed to the model verbatim (truncated) and relevance is the model's call.

Every search/fetch retries exactly once, then the caller degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

import httpx
import trafilatura
from ddgs import DDGS
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool

from app.config import settings
from app.llm.retry import call_with_retry

logger = logging.getLogger(__name__)

#: Injected into the generation prompts when research produced nothing (tool
#: failure, flag off, or the model chose not to search).
FALLBACK_RESEARCH_NOTES = "(no web research available)"

_NO_RESULTS = "No results."
_NO_CONTENT = "Could not extract content."

T = TypeVar("T")


def _retry_once(fn: Callable[[], T], *, what: str) -> T:
    """Run ``fn``; on exception log a warning and try exactly once more."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - retried once, then re-raised
        logger.warning("%s retrying after error: %s", what, exc)
        return fn()


def web_search(query: str) -> str:
    """Search DuckDuckGo; return 'title | url | snippet' lines."""
    def _run() -> list[dict]:
        return DDGS().text(query, max_results=settings.web_search_max_results)

    try:
        results = _retry_once(_run, what="web_search")
    except Exception as exc:  # noqa: BLE001 - final failure after one retry
        logger.error("web_search failed after retry: %s", exc)
        raise

    lines = [
        f"{r.get('title', '')} | {r.get('href', '')} | {r.get('body', '')}"
        for r in results
    ]
    logger.info("web_search query=%s results=%d", query, len(lines))
    return "\n".join(lines) if lines else _NO_RESULTS


def read_webpage(url: str) -> str:
    """Fetch ``url`` and return its extracted main text, truncated."""
    def _run() -> str:
        resp = httpx.get(
            url,
            timeout=settings.web_request_timeout_seconds,
            follow_redirects=True,
        )
        resp.raise_for_status()
        extracted = trafilatura.extract(resp.text) or ""
        return extracted[: settings.web_page_max_chars]

    try:
        text = _retry_once(_run, what="read_webpage")
    except Exception as exc:  # noqa: BLE001 - final failure after one retry
        logger.error("read_webpage failed after retry: %s", exc)
        raise

    text = text or _NO_CONTENT
    logger.info("read_webpage url=%s chars=%d", url, len(text))
    return text
