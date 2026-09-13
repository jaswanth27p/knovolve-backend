"""Web research for grounding LLM generation in live sources.

DuckDuckGo search (via ``ddgs``) discovers candidate URLs; page text is
extracted with ``trafilatura``. No embeddings or vector store: fetched text is
handed to the model verbatim (truncated) and relevance is the model's call.

Research is a single deterministic pass, not an agent tool loop: one search,
fetch the top N results once (in parallel), return the concatenated text. The
old LLM-driven tool loop was removed because it cost multiple model round-trips
per node and dominated course-creation wall time. Every fetch is best-effort;
a total failure degrades to no notes ("No results." / "").
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from ddgs import DDGS

from app.config import settings

logger = logging.getLogger(__name__)

#: Injected into the generation prompts when research produced nothing (flag
#: off, no results, or every fetch failed).
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


def web_search(query: str) -> str:
    """Search the web; return 'title | url | snippet' lines.

    Single pass, no whole-call retry: ddgs already falls through engines
    internally, and re-running the full fan-out after a total failure doubles
    a ~10-20s call. A total failure degrades to "No results." so the research
    pass moves on instead of stalling on repeated searches.
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


def _fetch_and_extract(url: str) -> str:
    """Fetch ``url`` (following redirects) and return its extracted main text.

    Raises on a blocked/non-public URL, a fetch failure, or too many redirects.
    """
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


def read_webpage_once(url: str) -> str:
    """Fetch ``url`` once (no retry); returns ``_NO_CONTENT`` on any failure.

    Used by the deterministic research pass, which must not add retry latency
    to the streaming chapter path.
    """
    try:
        text = _fetch_and_extract(url)
    except Exception as exc:  # noqa: BLE001 - best-effort single attempt
        logger.warning("read_webpage_once failed url=%s: %s", url, exc)
        return _NO_CONTENT
    return text or _NO_CONTENT


def _search_targets(search_output: str, limit: int) -> list[tuple[str, str]]:
    """Parse ``web_search`` lines into (title, url) pairs, capped at ``limit``."""
    targets: list[tuple[str, str]] = []
    for line in search_output.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 2 and parts[1].startswith("http"):
            targets.append((parts[0], parts[1]))
        if len(targets) >= limit:
            break
    return targets


def _one_shot_research(
    query: str,
    *,
    top_urls: int,
    max_chars_per_page: int,
    max_total_chars: int,
    label: str,
) -> str:
    """One ``web_search`` → fetch the top ``top_urls`` results once (no retry)
    in parallel → concatenate the extracted text as research notes. Returns
    ``""`` when nothing usable was gathered, so callers degrade to no notes."""
    try:
        search_output = web_search(query)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.warning("%s research search failed: %s", label, exc)
        return ""
    if not search_output or search_output == _NO_RESULTS:
        return ""

    targets = _search_targets(search_output, top_urls)
    if not targets:
        return ""

    with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as pool:
        texts = list(pool.map(lambda target: read_webpage_once(target[1]), targets))

    blocks: list[str] = []
    for (title, url), text in zip(targets, texts):
        if not text or text == _NO_CONTENT:
            continue
        blocks.append(f"Source: {title}\nURL: {url}\n{text[:max_chars_per_page]}")

    notes = "\n\n".join(blocks)[:max_total_chars].strip()
    logger.info(
        "%s research one-shot query=%s urls=%d blocks=%d notes_chars=%d",
        label, query, len(targets), len(blocks), len(notes),
    )
    return notes


def fetch_chapter_research(
    query: str,
    *,
    top_urls: int = 3,
    max_chars_per_page: int = 4000,
    max_total_chars: int = 10000,
) -> str:
    """Deterministic one-shot chapter research."""
    return _one_shot_research(
        query,
        top_urls=top_urls,
        max_chars_per_page=max_chars_per_page,
        max_total_chars=max_total_chars,
        label="chapter",
    )


def fetch_structure_research(
    query: str,
    *,
    top_urls: int | None = None,
    max_chars_per_page: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    """Deterministic one-shot research for course structure (outline/chapters).

    Honors ``web_search_enabled``; returns ``""`` when disabled or nothing
    usable was gathered so nodes fall back to ``(no web research available)``.
    """
    if not settings.web_search_enabled:
        return ""
    return _one_shot_research(
        query,
        top_urls=top_urls if top_urls is not None else settings.structure_research_top_urls,
        max_chars_per_page=(
            max_chars_per_page
            if max_chars_per_page is not None
            else settings.structure_research_max_chars_per_page
        ),
        max_total_chars=(
            max_total_chars
            if max_total_chars is not None
            else settings.structure_research_max_total_chars
        ),
        label="structure",
    )
