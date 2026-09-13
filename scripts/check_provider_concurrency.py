"""Empirically test whether the LLM provider parallelizes concurrent calls.

Runs the same tiny prompt at increasing concurrency (1, 2, 4, 8 by default)
and prints wall-clock time per level. Interpretation:

- If wall time stays roughly flat as concurrency rises, the provider accepts
  concurrent requests -> more Celery workers / section threads will help.
- If wall time scales ~linearly with concurrency, the provider serializes
  per key (or per x-opencode-session), so local parallelism buys nothing and
  throughput must come from reducing request count (batching).

Also runs one level with a UNIQUE x-opencode-session per call, to see whether
the shared process-lifetime session id is the thing being serialized.

Usage:
    cd backend
    .venv/bin/python scripts/check_provider_concurrency.py
    .venv/bin/python scripts/check_provider_concurrency.py --levels 1,2,4,8,16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app.config import settings  # noqa: E402
from app.llm.config import LLM_NODES  # noqa: E402
from app.llm.providers import PROVIDERS  # noqa: E402

PROMPT = (
    "Write a single concise paragraph of roughly 120 words explaining what a "
    "hash table is, aimed at a beginner. Output prose only."
)


def _client(session_id: str | None = None) -> ChatOpenAI:
    cfg = LLM_NODES["generate_chapter_section"]
    provider = PROVIDERS[cfg["provider"]]
    headers = {
        "User-Agent": "knovolve-concurrency-probe/1.0",
        "x-opencode-session": session_id or f"probe-{uuid.uuid4()}",
    }
    return ChatOpenAI(
        base_url=provider["base_url"],
        api_key=SecretStr(getattr(settings, provider["settings_field"])),
        model=cfg["model"],
        request_timeout=settings.llm_request_timeout_seconds,
        default_headers=headers,
    )


def _call(session_id: str | None) -> tuple[bool, str]:
    try:
        resp = _client(session_id).invoke(PROMPT)
        return True, str(resp.content)
    except Exception as exc:  # noqa: BLE001 - probe reports failures, not raises
        return False, f"{type(exc).__name__}: {exc}"


def _run_level(level: int, unique_session: bool) -> tuple[float, int]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=level) as pool:
        results = list(
            pool.map(
                lambda _: _call(f"probe-{uuid.uuid4()}" if unique_session else None),
                range(level),
            )
        )
    elapsed = time.perf_counter() - started
    ok = sum(1 for success, _ in results if success)
    if ok < level:
        for success, detail in results:
            if not success:
                print(f"    failure: {detail}")
    return elapsed, ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1,2,4,8")
    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    print(f"provider node: {LLM_NODES['generate_chapter_section']}, prompt ~120 words")
    print(f"{'level':>6} {'wall_s':>9} {'per_call_s':>11} {'ok':>4}  mode")
    baseline: float | None = None
    for level in levels:
        elapsed, ok = _run_level(level, unique_session=False)
        per_call = elapsed / level
        if level == 1:
            baseline = elapsed
        slowdown = f"{elapsed / baseline:.2f}x baseline" if baseline else ""
        print(f"{level:>6} {elapsed:>9.2f} {per_call:>11.2f} {ok:>4}  shared-session {slowdown}")

    print()
    for level in levels:
        if level == 1:
            continue
        elapsed, ok = _run_level(level, unique_session=True)
        per_call = elapsed / level
        slowdown = f"{elapsed / baseline:.2f}x baseline" if baseline else ""
        print(f"{level:>6} {elapsed:>9.2f} {per_call:>11.2f} {ok:>4}  unique-session {slowdown}")

    print()
    print("flat wall time while level rises -> provider parallel; linear -> serialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
