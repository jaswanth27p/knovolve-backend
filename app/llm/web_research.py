"""Web research tools for grounding LLM generation in live sources.

DuckDuckGo search (via ``ddgs``) discovers candidate URLs; page text is
extracted with ``trafilatura``. No embeddings or vector store: fetched text is
handed to the model verbatim (truncated) and relevance is the model's call.

Every fetch retries exactly once, then the caller degrades gracefully.
Search is a single pass over a pinned engine set (no whole-call retry —
re-running the fan-out would double a ~10-20s call); total search failure
degrades to "No results."
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from ddgs import DDGS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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

#: ddgs engines used for web search. google and brave aggressively rate-limit
#: automated queries (429s that stall each search ~10s+), and yahoo's
#: endpoint is TLS-flaky from this client — ddgs tries engines ~2 at a time
#: and early-stops once it has enough results, so a short reliable set beats
#: the full "auto" fan-out on latency with little relevance loss.
_SEARCH_BACKENDS = "duckduckgo,wikipedia,mojeek,startpage,grokipedia"

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
    """Search the web; return 'title | url | snippet' lines.

    Single pass, no whole-call retry: ddgs already falls through engines
    internally, and re-running the full fan-out after a total failure doubles
    a ~10-20s call. A total failure degrades to "No results." so the research
    loop moves on instead of stalling on repeated searches.
    """
    try:
        results = DDGS().text(
            query,
            max_results=settings.web_search_max_results,
            backend=_SEARCH_BACKENDS,
        )
    except Exception as exc:  # noqa: BLE001 - degraded, not fatal
        logger.warning("web_search failed fast: %s", exc)
        return _NO_RESULTS

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


def _content_text(resp: object) -> str:
    if isinstance(resp, AIMessage) and isinstance(resp.content, str):
        return resp.content
    return ""


def gather_research(
    model_with_tools: Runnable,
    messages: list[BaseMessage],
    max_rounds: int,
    max_tool_calls: int | None = None,
    summarizer: Runnable | None = None,
) -> str:
    """Run a bounded tool loop; return the model's final research brief.

    A tool failure after its internal retry becomes an error ``ToolMessage``
    and the loop continues (graceful degrade). When the round budget or the
    tool-call cap is exhausted and at least one tool result was gathered, a
    final unbound-model summarization pass (`summarizer`) turns the tool
    results into the brief. Returns ``""`` when nothing was gathered or no
    text could be produced.
    """
    impls: dict[str, Callable[..., str]] = {
        "web_search": web_search,
        "read_webpage": read_webpage,
    }
    rounds = 0
    tool_calls = 0

    def _finish() -> str:
        if summarizer is not None and tool_calls > 0:
            try:
                messages.append(HumanMessage(content=(
                    "Stop researching now and output your concise research notes "
                    "from the tool results above, with source URLs. Do not call any tools."
                )))
                resp = call_with_retry(summarizer.invoke, messages)
                content = _content_text(resp)
                if content:
                    logger.info(
                        "web research finished (summary) rounds=%d tool_calls=%d notes_chars=%d",
                        rounds, tool_calls, len(content),
                    )
                    return content
            except Exception as exc:  # noqa: BLE001 - best-effort summary
                logger.warning("web research summary failed: %s", exc)
        logger.warning("web research yielded no context")
        return ""

    for _ in range(max_rounds):
        resp = call_with_retry(model_with_tools.invoke, messages)
        if not isinstance(resp, AIMessage):
            break
        if not resp.tool_calls:
            content = _content_text(resp)
            logger.info(
                "web research finished rounds=%d tool_calls=%d notes_chars=%d",
                rounds, tool_calls, len(content),
            )
            if not content:
                return _finish()
            return content

        messages.append(resp)
        rounds += 1

        def _run(call: dict) -> tuple[str, str]:
            name = call["name"]
            try:
                result = impls[name](**call["args"])
            except KeyError:
                result = f"Error: no such tool '{name}'."
            except Exception as exc:  # noqa: BLE001 - model-facing recovery
                logger.warning("web tool %s failed: %s", name, exc)
                result = "Error: something went wrong calling this tool."
            return call["id"], str(result)

        def _flush(calls: list) -> None:
            if not calls:
                return
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                results = list(pool.map(_run, calls))
            for tool_call_id, content in results:
                messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

        pending = []
        for call in resp.tool_calls:
            if max_tool_calls is not None and tool_calls >= max_tool_calls:
                logger.warning("web research tool budget exhausted")
                _flush(pending)
                return _finish()
            tool_calls += 1
            logger.info("research round=%d tool_calls=%d", rounds, tool_calls)
            pending.append(call)
        _flush(pending)

    return _finish()


def run_web_research(
    model: Runnable,
    messages: list[BaseMessage],
    *,
    enabled: bool | None = None,
    max_rounds: int | None = None,
    max_tool_calls: int | None = None,
) -> str:
    """Feature-flagged entry point: research the given messages, or "" if off.

    `enabled`/`max_rounds`/`max_tool_calls` default to the course-structure
    settings; callers (chapter-content research) may override them to run
    under an independent flag and budget.
    """
    if enabled is None:
        enabled = settings.web_search_enabled
    if not enabled:
        return ""
    try:
        bound = model.bind_tools(build_web_tools())  # pyright: ignore[reportAttributeAccessIssue]
        return gather_research(
            bound,
            messages,
            max_rounds if max_rounds is not None else settings.web_research_max_tool_rounds,
            max_tool_calls if max_tool_calls is not None else settings.web_research_max_tool_calls,
            summarizer=model,
        )
    except Exception as exc:  # noqa: BLE001 - research is best-effort; degrade
        logger.warning("web research unavailable; continuing without it: %s", exc)
        return ""
