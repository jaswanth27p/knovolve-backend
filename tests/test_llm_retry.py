import pytest

from app.config import settings
from app.llm.retry import call_with_retry


def _fast_settings(monkeypatch, attempts=4):
    monkeypatch.setattr(settings, "llm_retry_max_attempts", attempts)
    monkeypatch.setattr(settings, "llm_retry_multiplier_seconds", 0.001)
    monkeypatch.setattr(settings, "llm_retry_max_delay_seconds", 0.01)
    monkeypatch.setattr(settings, "llm_retry_jitter_seconds", 0.0)


def test_transient_error_retries_then_succeeds(monkeypatch):
    _fast_settings(monkeypatch)
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("upstream unreachable")
        return "ok"

    assert call_with_retry(flaky) == "ok"
    assert attempts == 3


def test_exhausts_budget_and_reraises(monkeypatch):
    _fast_settings(monkeypatch, attempts=2)
    attempts = 0

    def always_fails():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("still down")

    with pytest.raises(TimeoutError):
        call_with_retry(always_fails)
    assert attempts == 2


def test_non_retryable_error_propagates_immediately(monkeypatch):
    """Anything not on the retryable list (e.g. a 4xx validation error) must
    not be retried: the failure is deterministic and retrying wastes budget."""

    def bad_input():
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        call_with_retry(bad_input)