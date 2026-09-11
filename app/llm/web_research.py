"""Web research tools for grounding LLM generation in live sources.

DuckDuckGo search (via ``ddgs``) discovers candidate URLs; page text is
extracted with ``trafilatura``. No embeddings or vector store: fetched text is
handed to the model verbatim (truncated) and relevance is the model's call.

Every search/fetch retries exactly once, then the caller degrades gracefully.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Callable, TypeVar
from urllib.parse import urljoin, urlsplit

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

_MAX_REDIRECTS = 5

T = TypeVar("T")


def _assert_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts resolving to private/loopback/
    link-local/reserved/multicast/unspecified addresses (SSRF guard)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"blocked URL scheme: {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise ValueError("URL has no host")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        addrs = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host {host!r}") from exc
    for _family, _type, _proto, _canon, sockaddr in addrs:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"blocked non-public address for {host!r}: {ip}")


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
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            _assert_public_url(current)
            resp = httpx.get(
                current,
                timeout=settings.web_request_timeout_seconds,
                follow_redirects=False,
            )
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    break
                current = urljoin(current, location)
                continue
            resp.raise_for_status()
            extracted = trafilatura.extract(resp.text) or ""
            return extracted[: settings.web_page_max_chars]
        raise ValueError("too many redirects")

    try:
        text = _retry_once(_run, what="read_webpage")
    except Exception as exc:  # noqa: BLE001 - final failure after one retry
        logger.error("read_webpage failed after retry: %s", exc)
        raise

    text = text or _NO_CONTENT
    logger.info("read_webpage url=%s chars=%d", url, len(text))
    return text


def build_web_tools() -> list[BaseTool]:
    """LangChain tools for the research phase."""

    @tool("web_search")
    def _web_search(query: str) -> str:
        """Search the web with DuckDuckGo. Input: a search query. Returns
        result lines formatted as 'title | url | snippet'."""
        return web_search(query)

    @tool("read_webpage")
    def _read_webpage(url: str) -> str:
        """Fetch a web page and return its main text content. Input: a full
        URL including the https:// scheme."""
        return read_webpage(url)

    return [_web_search, _read_webpage]


def gather_research(
    model_with_tools: Runnable,
    messages: list[BaseMessage],
    max_rounds: int,
    max_tool_calls: int | None = None,
) -> str:
    """Run a bounded tool loop; return the model's final research brief.

    A tool failure after its internal retry becomes an error ``ToolMessage``
    and the loop continues (graceful degrade). Returns ``""`` when the budget
    is exhausted or the model never produces text.
    """
    impls: dict[str, Callable[..., str]] = {
        "web_search": web_search,
        "read_webpage": read_webpage,
    }
    rounds = 0
    tool_calls = 0
    for _ in range(max_rounds):
        resp = call_with_retry(model_with_tools.invoke, messages)
        if not isinstance(resp, AIMessage):
            break
        if not resp.tool_calls:
            content = resp.content if isinstance(resp.content, str) else ""
            logger.info(
                "web research finished rounds=%d tool_calls=%d notes_chars=%d",
                rounds, tool_calls, len(content),
            )
            if not content:
                logger.warning("web research yielded no context")
            return content

        messages.append(resp)
        rounds += 1
        for call in resp.tool_calls:
            if max_tool_calls is not None and tool_calls >= max_tool_calls:
                logger.warning("web research tool budget exhausted")
                return ""
            tool_calls += 1
            name = call["name"]
            logger.info("research round=%d tool_calls=%d", rounds, tool_calls)
            try:
                result = impls[name](**call["args"])
            except KeyError:
                result = f"Error: no such tool '{name}'."
            except Exception as exc:  # noqa: BLE001 - model-facing recovery
                logger.warning("web tool %s failed: %s", name, exc)
                result = "Error: something went wrong calling this tool."
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    logger.warning("web research yielded no context")
    return ""


def run_web_research(model: Runnable, messages: list[BaseMessage]) -> str:
    """Feature-flagged entry point: research the given messages, or "" if off."""
    if not settings.web_search_enabled:
        return ""
    try:
        bound = model.bind_tools(build_web_tools())  # pyright: ignore[reportAttributeAccessIssue]
        return gather_research(
            bound,
            messages,
            settings.web_research_max_tool_rounds,
            settings.web_research_max_tool_calls,
        )
    except Exception as exc:  # noqa: BLE001 - research is best-effort; degrade
        logger.warning("web research unavailable; continuing without it: %s", exc)
        return ""
