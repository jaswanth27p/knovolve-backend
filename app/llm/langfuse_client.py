"""Langfuse observability wiring. Mirrors app/observability.py's otel_enabled
pattern: every entry point here is a no-op when langfuse_enabled is False, so
disabling it is a one-line env var flip with zero behavior change elsewhere.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import os
from typing import Any, Callable, Generator, Iterator, ParamSpec, TypeVar, cast

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


def flush_langfuse() -> None:
    """Force-flush buffered spans before a short-lived process (a Celery
    task's worker child, which may be recycled via
    celery_worker_max_tasks_per_child) exits — the batch exporter's
    background thread does not guarantee delivery before that happens.
    No-op when disabled. Call from a Celery task wrapper's `finally` block
    (app/tasks/*.py), not from the traced_workflow-wrapped function itself —
    HTTP-request-scoped workflows (chat, chapter content generation, custom
    export clarify) run in the long-lived FastAPI process and don't need
    this; only Celery-task-scoped workflows do."""
    if not settings.langfuse_enabled:
        return
    from langfuse import get_client

    get_client().flush()


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


@contextlib.contextmanager
def _context_applied(ctx: contextvars.Context) -> Iterator[None]:
    """Copy every value held by `ctx` onto whatever context is current for the
    duration of the block, then restore the current context exactly as it was.

    This is the single re-apply mechanism shared by `run_with_current_context`
    (thread-pool workers) and `traced_workflow_stream` (generator resumptions).
    It deliberately does NOT use `contextvars.Context.run(...)`: re-running one
    captured `Context` object raises `RuntimeError: cannot enter context: ...
    is already entered` as soon as two threads do it at once (the bug fixed in
    commit 5aa8a1a). `ContextVar.set()`/`.reset()` writes into the *caller's
    own* implicit context instead, so N threads re-applying the same captured
    snapshot concurrently is safe, and so is re-applying it from a different
    context on every single generator step."""
    tokens = [(var, var.set(value)) for var, value in ctx.items()]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def traced_workflow_stream(
    source: Iterator[T],
    trace_name: str,
    *,
    user_id: str | int | None = None,
    session_id: str | int | None = None,
    tags: list[str] | None = None,
) -> Generator[T, None, None]:
    """`traced_workflow(...)` for a generator that is consumed one item at a
    time by an async server — use this instead of the plain
    `with traced_workflow(...): yield from _inner(...)` shape whenever the
    result is handed to a `StreamingResponse`.

    Why the `with`-around-`yield from` shape is broken: Starlette drives a sync
    iterator through `iterate_in_threadpool`, which calls
    `anyio.to_thread.run_sync(_next, iterator)` ONCE PER ITEM. anyio's asyncio
    backend does a fresh `copy_context()` for every one of those calls, copying
    from the *event loop* thread — not from the context the previous `next()`
    ran in. `traced_workflow`'s `propagate_attributes` is an OTel
    `context.attach()`, i.e. a `ContextVar.set()`; it happens on the
    generator's FIRST resumption, inside a throwaway per-item context copy, and
    is therefore gone before the second `next()` is ever made. Every LLM call
    after the first yielded event then starts a brand-new root trace with no
    `trace_name`/`user_id`/`tags`.

    The fix: open the trace once, snapshot the resulting context, and re-apply
    that snapshot around EVERY step of the wrapped generator. The snapshot is
    also refreshed at the end of each step, so context established *inside* the
    generator (e.g. a langchain span that stays open across several streamed
    chunks) is threaded into the next step too, rather than being discarded
    back to the state at trace-open time.

    The context is applied only while a step is actually running: it is
    restored before each `yield`, so the consumer is never left holding the
    generator's contextvars while suspended.

    No-op wrapper when Langfuse is disabled — `traced_workflow` returns a noop
    contextmanager and the snapshot re-apply is then just a few dict writes."""
    with contextlib.ExitStack() as stack:
        stack.callback(_close_quietly, source)
        stack.enter_context(
            traced_workflow(trace_name, user_id=user_id, session_id=session_id, tags=tags)
        )
        # Captured INSIDE the trace context manager, on the same resumption
        # that entered it — that is the only moment the attached OTel context
        # is reliably visible.
        snapshot = contextvars.copy_context()
        while True:
            with _context_applied(snapshot):
                try:
                    item = next(source)
                except StopIteration:
                    return
                finally:
                    # Re-snapshot before the tokens are reset so anything the
                    # step attached survives into the following step.
                    snapshot = contextvars.copy_context()
            yield item


def _close_quietly(source: object) -> None:
    """Close the wrapped generator when the outer stream is abandoned or
    fails, so it can run its own `finally` blocks promptly instead of waiting
    for the GC. `Iterator` has no `close()`, hence the guard."""
    close = getattr(source, "close", None)
    if callable(close):
        close()


def run_with_current_context(fn: Callable[P, T]) -> Callable[P, T]:
    """Wrap `fn` so a ThreadPoolExecutor worker thread sees the SAME
    contextvars (and therefore the same open Langfuse/OTel trace) as the
    thread that submitted it. Plain `executor.submit(fn, ...)` does NOT do
    this: a new OS thread starts with an empty Context, not the submitting
    thread's — the identical fix OTel's own thread-pool instrumentation
    needs. Call this on the submitting thread, right before `.submit()`/
    `.map()`, so `copy_context()` captures the trace that's actually open.

    The returned wrapper must tolerate being invoked concurrently from
    several worker threads at once (that's exactly what `executor.map(...)`
    does when called with a single wrapped callable across many items) —
    it does NOT re-run the captured `Context` object itself via
    `ctx.run(...)`. `contextvars.Context.run()` is documented to raise
    `RuntimeError: cannot enter context: ... is already entered` when the
    SAME Context object is entered from more than one OS thread at once,
    which one wrapped call sharing one `ctx` across a thread pool triggers
    reliably. Instead, each invocation copies the captured vars' values onto
    its own (per-thread) context via `ContextVar.set()`/`.reset()`, which is
    safe to do concurrently since every thread has its own implicit
    Context."""
    ctx = contextvars.copy_context()

    @functools.wraps(fn)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        with _context_applied(ctx):
            return fn(*args, **kwargs)

    return _wrapped
