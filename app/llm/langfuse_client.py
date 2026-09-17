"""Langfuse observability wiring. Mirrors app/observability.py's otel_enabled
pattern: every entry point here is a no-op when langfuse_enabled is False, so
disabling it is a one-line env var flip with zero behavior change elsewhere.
"""
from __future__ import annotations

import contextvars
import functools
import os
from typing import Any, Callable, ParamSpec, TypeVar, cast

from app.config import settings

P = ParamSpec("P")
T = TypeVar("T")

_handler = None
_handler_built = False


def setup_langfuse() -> None:
    """Call once at process startup (FastAPI app import, Celery worker
    import). No-ops when disabled. Sets the LANGFUSE_* env vars the SDK
    reads internally from `settings`, matching how OTEL_* env vars can
    override observability.py's own settings-backed config."""
    if not settings.langfuse_enabled:
        return
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", settings.langfuse_tracing_environment)
    from langfuse import get_client

    get_client()  # warms the client; never raises for an unreachable host


def get_langfuse_handler():
    """Cached singleton CallbackHandler, or None when disabled. One shared
    instance is bound at every app/llm/factory.py call site — see Task 6."""
    global _handler, _handler_built
    if not settings.langfuse_enabled:
        return None
    if _handler_built:
        return _handler
    from langfuse.langchain import CallbackHandler

    _handler = CallbackHandler()
    _handler_built = True
    return _handler


class _NoopWorkflow:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> bool:
        return False


def traced_workflow(
    trace_name: str,
    *,
    user_id: str | int | None = None,
    session_id: str | int | None = None,
    tags: list[str] | None = None,
):
    """Wrap one workflow's top-level entry point in this. Every LLM call made
    synchronously inside the block — across however many node functions or
    tool-call rounds — lands in the SAME Langfuse trace, because tracing rides
    on OTel contextvars. No-op contextmanager when Langfuse is disabled."""
    if not settings.langfuse_enabled:
        return _NoopWorkflow()
    from langfuse import propagate_attributes

    kwargs: dict[str, object] = {"trace_name": trace_name, "tags": tags or []}
    if user_id is not None:
        kwargs["user_id"] = str(user_id)
    if session_id is not None:
        kwargs["session_id"] = str(session_id)
    return propagate_attributes(**cast(dict[str, Any], kwargs))


def run_with_current_context(fn: Callable[P, T]) -> Callable[P, T]:
    """Wrap `fn` so a ThreadPoolExecutor worker thread sees the SAME
    contextvars (and therefore the same open Langfuse/OTel trace) as the
    thread that submitted it. Plain `executor.submit(fn, ...)` does NOT do
    this: a new OS thread starts with an empty Context, not the submitting
    thread's — the identical fix OTel's own thread-pool instrumentation
    needs. Call this on the submitting thread, right before `.submit()`/
    `.map()`, so `copy_context()` captures the trace that's actually open."""
    ctx = contextvars.copy_context()

    @functools.wraps(fn)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        return ctx.run(fn, *args, **kwargs)

    return _wrapped
