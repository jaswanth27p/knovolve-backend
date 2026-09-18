import contextvars

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
