import logging
import os

from app.config import settings

_INITIALIZED = False


def _env_or(name, default: str) -> str:
    return os.environ.get(name, default) or default


def otel_enabled() -> bool:
    # Explicit env var can override the .env-backed setting (useful for
    # one-off local runs and for the containerized deployment).
    return _env_or("OTEL_ENABLED", "1" if settings.otel_enabled else "0").lower() in ("1", "true", "yes")


def _otlp_base() -> str:
    base = settings.otel_exporter_otlp_endpoint
    # Strip any trailing slash so we can append signal paths cleanly.
    return base.rstrip("/")


def setup_tracing() -> None:
    if not otel_enabled():
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.otel_service_name,
            }
        )
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{_otlp_base()}/v1/traces")))
    set_tracer_provider(provider)


def set_tracer_provider(provider) -> None:
    from opentelemetry import trace

    trace.set_tracer_provider(provider)


def setup_metrics(app) -> None:
    if not otel_enabled():
        return
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=False,
        should_group_untemplated=True,
        should_ignore_untemplated=True,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


class OTLPLoggingHandler(logging.Handler):
    """Custom bridge from stdlib logging into the OTel logs pipeline.

    The SDK's LoggingHandler has proven unreliable here (records silently
    dropped), so we translate each stdlib record into an OTel LogRecord and
    emit through the LoggerProvider directly — the path verified end-to-end.
    """

    def __init__(self, level: int, provider) -> None:
        super().__init__(level=level)
        self._provider = provider

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._provider.get_logger(record.name).emit(
                body=record.getMessage(),
                severity_text=record.levelname,
                attributes={
                    "code.filepath": record.pathname,
                    "code.function": record.funcName,
                    "code.lineno": record.lineno,
                },
            )
        except Exception:
            self.handleError(record)


_NOISY_ACCESS_PATHS = ("/health", "/metrics")


class _AccessLogNoiseFilter(logging.Filter):
    """Drop uvicorn access-log lines for high-frequency probe endpoints
    (health checks, metrics scrapes) so they don't flood Loki."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            path = str(args[2])
            if any(path == p or path.startswith(f"{p}?") for p in _NOISY_ACCESS_PATHS):
                return False
        return True


_otlp_handler: "OTLPLoggingHandler | None" = None


def _get_or_create_otlp_handler() -> "OTLPLoggingHandler | None":
    """Build (once) and return the process-wide OTLP logging handler.

    Cached at module scope so every caller -- setup_logging() itself, and
    the celery after_setup_logger/after_setup_task_logger signal handlers in
    app/tasks/celery_app.py -- shares one LoggerProvider/exporter instead of
    opening a duplicate OTLP connection per logger bridged.
    """
    global _otlp_handler
    if not otel_enabled():
        return None
    if _otlp_handler is not None:
        return _otlp_handler
    import logging as _logging

    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    provider = LoggerProvider(
        resource=Resource.create(
            {
                "service.name": settings.otel_service_name,
            }
        )
    )
    provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{_otlp_base()}/v1/logs")))
    level = getattr(_logging, settings.otel_log_level.upper(), _logging.INFO)
    _otlp_handler = OTLPLoggingHandler(level=level, provider=provider)
    return _otlp_handler


def attach_otlp_handler(logger: logging.Logger) -> None:
    """Attach the shared OTLP handler to `logger`. No-op if OTel is disabled.

    Celery reconfigures its own logger tree (root included, unless
    worker_hijack_root_logger=False) during worker/beat bootstrap, which runs
    well *after* this module is imported -- so a handler attached at import
    time gets silently dropped. Call this from celery's after_setup_logger /
    after_setup_task_logger signals instead, which fire once Celery is done,
    so the bridge actually survives.
    """
    handler = _get_or_create_otlp_handler()
    if handler is not None and not any(isinstance(h, OTLPLoggingHandler) for h in logger.handlers):
        logger.addHandler(handler)


def setup_logging() -> None:
    if not otel_enabled():
        return
    import logging as _logging

    handler = _get_or_create_otlp_handler()
    assert handler is not None
    root = _logging.getLogger()
    # Python's root logger defaults to WARNING, which would drop INFO records
    # before OTLP export. Lift it so the configured level reaches handlers.
    root.setLevel(handler.level)
    # Don't double-attach if setup is called again in-process (uvicorn reload).
    if not any(isinstance(h, OTLPLoggingHandler) for h in root.handlers):
        root.addHandler(handler)
    # uvicorn configures its loggers with propagate=False, so root never sees
    # them. Attach the same bridge directly so request/access lines reach
    # OTLP too. Celery's loggers are bridged separately via the
    # after_setup_logger/after_setup_task_logger signals in celery_app.py --
    # see attach_otlp_handler()'s docstring for why they can't be attached
    # here at import time.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        attach_otlp_handler(_logging.getLogger(name))

    access_logger = _logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _AccessLogNoiseFilter) for f in access_logger.filters):
        access_logger.addFilter(_AccessLogNoiseFilter())

    # warnings.warn() (e.g. PyJWT's InsecureKeyLengthWarning, celery's
    # SecurityWarning about running as root) goes straight to stderr via the
    # warnings module by default, bypassing logging entirely. Route it
    # through the "py.warnings" logger instead so it reaches OTLP too.
    _logging.captureWarnings(True)
    warnings_logger = _logging.getLogger("py.warnings")
    attach_otlp_handler(warnings_logger)
    # Unlike uvicorn's loggers (which set propagate=False themselves),
    # py.warnings defaults to propagate=True -- without this it would also
    # hit root's copy of the same handler on the way up, double-emitting
    # every captured warning.
    warnings_logger.propagate = False


def instrument_fastapi(app) -> None:
    if not otel_enabled():
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    # Same probe endpoints as _AccessLogNoiseFilter above, so a health check
    # or metrics scrape doesn't also spam Tempo with a span every 10-15s.
    # excluded_urls is matched as a regex against the full request URL, so
    # this also needs no other route to contain "/health" or "/metrics" as a
    # substring -- true today, worth rechecking if that ever changes.
    FastAPIInstrumentor.instrument_app(app, excluded_urls=",".join(_NOISY_ACCESS_PATHS))


def instrument_static() -> None:
    """Instrument libs that are not tied to the FastAPI app (requests, httpx,
    redis, sqlalchemy). Safe to call once from both web and celery entrypoints."""
    global _INITIALIZED
    if not otel_enabled() or _INITIALIZED:
        return
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    from app.db import engine

    RequestsInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    RedisInstrumentor().instrument()
    SQLAlchemyInstrumentor().instrument(engine=engine)
    CeleryInstrumentor().instrument()
    _INITIALIZED = True


def setup_sentry() -> None:
    dsn = settings.sentry_dsn or _env_or("SENTRY_DSN", "")
    if not dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        environment=_env_or("KNOVOLVE_ENV", "development"),
        send_default_pii=False,
    )