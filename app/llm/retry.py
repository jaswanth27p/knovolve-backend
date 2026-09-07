"""Transient-failure retry for external LLM/embedding calls.

A single provider timeout, 429, or 5xx must not abort a multi-minute course
generation run. Every external call in the course-creation workflow is routed
through :func:`call_with_retry`, which retries *only* the transient exception
classes below with exponential backoff + jitter and re-raises everything else
immediately so non-retryable failures surface unchanged.

The policy reads its knobs from ``Settings`` (``llm_retry_*``). They are
workflow-scoped today by virtue of being read here; moving the same settings to
a global policy later requires no call-site changes.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from app.config import settings

T = TypeVar("T")

#: Everything else (4xx other than 429, validation errors, programming bugs)
#: is deliberately NOT retryable: retrying a deterministic failure just burns
#: budget. Openai's APITimeoutError is a subclass of APIConnectionError, so
#: listing both is harmless but explicit.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    ConnectionError,
    TimeoutError,
)

_logger = logging.getLogger(__name__)


def call_with_retry(fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Call ``fn(*args, **kwargs)``, retrying transient errors with backoff.

    A fresh decorated wrapper is built per call so attempt counters never leak
    across unrelated invocations (tenacity 9 removed the reusable
    ``Retrying.call`` entry point).
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.llm_retry_max_attempts),
        wait=wait_exponential(
            multiplier=settings.llm_retry_multiplier_seconds,
            max=settings.llm_retry_max_delay_seconds,
            exp_base=2,
        )
        + wait_random(0, settings.llm_retry_jitter_seconds),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    def _inner(*a: object, **k: object) -> T:
        return fn(*a, **k)

    return _inner(*args, **kwargs)
