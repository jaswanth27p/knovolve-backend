import contextlib
import contextvars

import pytest

from app.config import settings
from app.llm.langfuse_client import (
    flush_langfuse,
    get_langfuse_handler,
    run_with_current_context,
    traced_workflow,
)


def test_handler_none_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    assert get_langfuse_handler() is None


def test_flush_langfuse_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    flush_langfuse()  # must not raise, must not import langfuse


def test_traced_workflow_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    entered = False
    with traced_workflow("Test Workflow", user_id=1, session_id="s1", tags=["x"]):
        entered = True
    assert entered


_probe: contextvars.ContextVar[str] = contextvars.ContextVar("probe", default="unset")


def test_run_with_current_context_propagates_contextvar():
    """The whole point of this helper: a value set on the submitting thread
    must be visible inside the wrapped callable even when a ThreadPoolExecutor
    runs it on a different OS thread — plain executor.submit does NOT do this
    by default."""
    from concurrent.futures import ThreadPoolExecutor

    _probe.set("main-thread-value")

    def read_probe():
        return _probe.get()

    wrapped = run_with_current_context(read_probe)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(wrapped).result()

    assert result == "main-thread-value"


def test_run_with_current_context_safe_under_concurrent_invocation():
    """Regression test for a real bug found by Task 17's full-suite run:
    when the SAME wrapped callable is invoked concurrently from several
    worker threads at once (exactly what `ThreadPoolExecutor.map(wrapped,
    items)` does — one `run_with_current_context(...)` call, many items),
    the naive `ctx.run(fn, ...)` implementation raised `RuntimeError:
    cannot enter context: ... is already entered`, because
    contextvars.Context.run() is documented to disallow entering the same
    Context object from more than one OS thread concurrently. This is what
    app/agents/assignment/generate.py and
    app/agents/chapter_content/generate.py actually do (one wrapped
    function, executor.map over N sections/slots), so the bug was real and
    reachable, not just theoretical."""
    from concurrent.futures import ThreadPoolExecutor
    import time

    _probe.set("concurrent-value")

    def read_probe(n: int) -> tuple[int, str]:
        # Sleep so multiple worker threads are guaranteed to overlap inside
        # the wrapped call at the same time, reproducing the race.
        time.sleep(0.05)
        return n, _probe.get()

    wrapped = run_with_current_context(read_probe)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(wrapped, range(8)))

    assert results == [(n, "concurrent-value") for n in range(8)]


def test_plain_submit_without_wrapper_does_not_propagate():
    """Sanity check that the fix in the test above is actually doing
    something, not passing by accident."""
    from concurrent.futures import ThreadPoolExecutor

    _probe.set("main-thread-value-2")

    def read_probe():
        return _probe.get()

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(read_probe).result()

    assert result == "unset"


def test_handler_is_singleton_when_enabled(monkeypatch):
    """Verify that get_langfuse_handler() returns the same object on
    repeated calls when langfuse_enabled=True, and CallbackHandler is
    only constructed once. This exercises the _handler_built caching
    logic that would silently break if refactored incorrectly."""
    from unittest.mock import MagicMock, patch
    import app.llm.langfuse_client as lc

    monkeypatch.setattr(settings, "langfuse_enabled", True)
    # Reset the global state to simulate a fresh import
    monkeypatch.setattr(lc, "_handler_built", False)
    monkeypatch.setattr(lc, "_handler", None)

    fake_handler = MagicMock()
    with patch("langfuse.langchain.CallbackHandler", return_value=fake_handler) as mock_ctor:
        first = get_langfuse_handler()
        second = get_langfuse_handler()

        assert first is second, "Handler should be the same object on repeated calls"
        assert first is fake_handler, "Handler should be the mocked instance"
        mock_ctor.assert_called_once()


# --- traced_workflow_stream: streaming/generator context propagation ---------
#
# Regression coverage for the bug the final whole-branch review found: a
# `with traced_workflow(...): yield from _inner(...)` generator loses its trace
# after the FIRST yielded item when Starlette drives it. Starlette's
# StreamingResponse consumes a sync iterator via `iterate_in_threadpool`, which
# calls `anyio.to_thread.run_sync(_next, it)` once per item, and anyio's
# asyncio backend does a fresh `copy_context()` for EVERY one of those calls —
# copied from the event-loop thread, not carried over from the previous step.
# So the `ContextVar.set()` that `propagate_attributes` performs on the first
# resumption is discarded before the second `next()` happens.
#
# These tests therefore drive the generator through the REAL
# `iterate_in_threadpool` path. A `list(...)` based test cannot catch this bug
# (see test_plain_with_yield_from_shape_loses_trace_across_resumptions, which
# fails the same assertion only because it goes through the real path).

_trace_marker: contextvars.ContextVar[str] = contextvars.ContextVar("trace_marker", default="no-trace")

#: Records cross-context `ContextVar.reset()` failures, exactly like langfuse's
#: own `_detach_context_token_safely` swallows them.
_stale_detaches: list[str] = []


@contextlib.contextmanager
def _fake_trace(trace_name: str, **_kwargs):
    """Faithful stand-in for `traced_workflow`. Real `propagate_attributes` is
    an OTel `context.attach(...)` — i.e. a single `ContextVar.set()` — plus a
    detach on exit that swallows the `ValueError` raised when the exit happens
    in a different context than the enter (langfuse's
    `_detach_context_token_safely`). Mirroring that swallow matters: the
    generator's last step genuinely runs in a different context than its
    first."""
    token = _trace_marker.set(trace_name)
    try:
        yield
    finally:
        try:
            _trace_marker.reset(token)
        except ValueError:
            _stale_detaches.append(trace_name)


def _drive_like_starlette(iterator):
    """Consume `iterator` the way a live HTTP request does: StreamingResponse
    -> `starlette.concurrency.iterate_in_threadpool` -> `anyio.to_thread.
    run_sync`, which copies a FRESH context per item off the event loop."""
    import anyio
    from starlette.concurrency import iterate_in_threadpool

    async def _consume():
        return [item async for item in iterate_in_threadpool(iterator)]

    return anyio.run(_consume)


def test_traced_workflow_stream_keeps_trace_across_threadpool_resumptions(monkeypatch):
    """THE regression test for finding #1. The trace opened once by
    traced_workflow_stream must still be attached on every later resumption of
    the wrapped generator, not just the first."""
    import app.llm.langfuse_client as lc

    monkeypatch.setattr(lc, "traced_workflow", _fake_trace)
    seen: list[str] = []

    def _events():
        for i in range(4):
            seen.append(_trace_marker.get())
            yield {"i": i}

    items = _drive_like_starlette(lc.traced_workflow_stream(_events(), "Chat Reply"))

    assert items == [{"i": i} for i in range(4)]
    assert seen == ["Chat Reply"] * 4, (
        "the trace must survive every resumption, not only the first"
    )


def test_plain_with_yield_from_shape_loses_trace_across_resumptions(monkeypatch):
    """Pins the bug itself, so the test above is proven to have teeth: the old
    `with traced_workflow(...): yield from ...` shape really does lose the
    trace after the first item when driven through the real Starlette path."""
    seen: list[str] = []

    def _old_shape():
        with _fake_trace("Chat Reply"):
            for i in range(4):
                seen.append(_trace_marker.get())
                yield {"i": i}

    _drive_like_starlette(_old_shape())

    assert seen[0] == "Chat Reply"
    assert seen[1:] == ["no-trace"] * 3, (
        "if this ever passes as ['Chat Reply'] * 4, anyio stopped copying a "
        "fresh context per item and the primitive's raison d'etre changed"
    )


def test_traced_workflow_stream_propagates_context_established_inside_the_generator(monkeypatch):
    """Context attached by the generator's own body (e.g. a langchain/langfuse
    span left open across several streamed chunks) is threaded into the next
    step too, not reset back to the trace-open snapshot."""
    import app.llm.langfuse_client as lc

    monkeypatch.setattr(lc, "traced_workflow", _fake_trace)
    inner: contextvars.ContextVar[str] = contextvars.ContextVar("inner", default="unset")
    seen: list[str] = []

    def _events():
        yield {"n": 0}
        inner.set("span-open")
        for i in range(1, 4):
            seen.append(inner.get())
            yield {"n": i}

    _drive_like_starlette(lc.traced_workflow_stream(_events(), "Chat Reply"))

    assert seen == ["span-open"] * 3


def test_traced_workflow_stream_propagates_exceptions_and_exits_trace(monkeypatch):
    """A raising generator must surface the exception unchanged, and the trace
    context manager must still be exited (no leaked span)."""
    import app.llm.langfuse_client as lc

    exits: list[str] = []

    @contextlib.contextmanager
    def _recording_trace(trace_name: str, **_kwargs):
        token = _trace_marker.set(trace_name)
        try:
            yield
        finally:
            exits.append(trace_name)
            try:
                _trace_marker.reset(token)
            except ValueError:
                pass

    monkeypatch.setattr(lc, "traced_workflow", _recording_trace)

    class Boom(RuntimeError):
        pass

    def _events():
        yield {"n": 0}
        raise Boom("inner exploded")

    stream = lc.traced_workflow_stream(_events(), "Chat Reply")
    assert next(stream) == {"n": 0}
    with pytest.raises(Boom, match="inner exploded"):
        next(stream)
    assert exits == ["Chat Reply"], "trace must be exited exactly once on failure"


def test_traced_workflow_stream_closes_inner_generator_when_abandoned(monkeypatch):
    """Abandoning the stream (client disconnect) must close the wrapped
    generator so its own `finally` blocks run, and must exit the trace."""
    import app.llm.langfuse_client as lc

    monkeypatch.setattr(lc, "traced_workflow", _fake_trace)
    closed: list[bool] = []

    def _events():
        try:
            for i in range(10):
                yield {"n": i}
        finally:
            closed.append(True)

    stream = lc.traced_workflow_stream(_events(), "Chat Reply")
    assert next(stream) == {"n": 0}
    stream.close()

    assert closed == [True]


def test_traced_workflow_stream_is_transparent_when_langfuse_disabled(monkeypatch):
    """With Langfuse off, the wrapper is a pass-through: same items, same
    order, no import of langfuse."""
    import app.llm.langfuse_client as lc

    monkeypatch.setattr(settings, "langfuse_enabled", False)

    def _events():
        yield from [{"n": 0}, {"n": 1}, {"n": 2}]

    assert list(lc.traced_workflow_stream(_events(), "Chat Reply", user_id=7)) == [
        {"n": 0}, {"n": 1}, {"n": 2},
    ]
