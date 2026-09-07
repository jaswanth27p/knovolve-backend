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


def setup_logging() -> None:
    if not otel_enabled():
        return
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
    root = _logging.getLogger()
    # Python's root logger defaults to WARNING, which would drop INFO records
    # before OTLP export. Lift it so the configured level reaches handlers.
    root.setLevel(level)
    handler = OTLPLoggingHandler(level=level, provider=provider)
    # Don't double-attach if setup is called again in-process (uvicorn reload).
    if not any(isinstance(h, OTLPLoggingHandler) for h in root.handlers):
        root.addHandler(handler)
    # uvicorn and celery configure their loggers with propagate=False, so root
    # never sees them. Attach the same bridge directly so request/access lines
    # and celery task logs reach OTLP too.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "celery", "celery.task"):
        lgr = _logging.getLogger(name)
        if not any(isinstance(h, OTLPLoggingHandler) for h in lgr.handlers):
            lgr.addHandler(handler)


def instrument_fastapi(app) -> None:
    if not otel_enabled():
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


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