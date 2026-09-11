from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://knovolve:knovolve@localhost:5432/knovolve"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str
    jwt_access_ttl_minutes: int = 15
    jwt_refresh_ttl_days: int = 30
    # Hard cap on a login session's lifetime regardless of activity. A chain
    # of rotated refresh tokens shares one session_started_at (its birth);
    # once the chain is older than this, /auth/refresh refuses to rotate any
    # further and the user must log in again. Without this, a continuously
    # active user would refresh forever and never re-authenticate. Set to 0
    # to disable. Idle users are already bounded by jwt_refresh_ttl_days on
    # each token; this caps the active-user case.
    force_relogin_after_days: int = 30
    # How long a revoked/expired refresh_tokens row is kept before the
    # cleanup task purges it. Kept briefly post-revoke for reuse-detection
    # forensics rather than deleted immediately.
    refresh_token_purge_after_days: int = 30
    opencode_api_key: str
    opencode_zen_api_key: str = ""
    openrouter_api_key: str
    # JSON list in .env, e.g. CORS_ORIGINS=["http://localhost:3000"]
    cors_origins: list[str] = ["http://localhost:3000"]
    # SameSite policy for the auth cookies. "strict" is the secure default for
    # same-origin dev; set "none" in production when the frontend is served
    # from a different origin than the API (e.g. Next.js on Vercel, API on
    # Render/Railway/Fly) — SameSite=Strict cookies are never attached to the
    # cross-site API calls, silently breaking auth. "none" requires the
    # cookies to also be Secure (they are) and is only safe over HTTPS.
    cookie_samesite: Literal["lax", "strict", "none"] = "strict"
    # Base URL the frontend is served from — used to build clickable
    # course_url links in chat-agent tool responses.
    frontend_url: str = "http://localhost:3000"
    # Minimum cosine similarity for a Course/CourseJob to count as an existing
    # version of a requested topic (dedup).
    topic_similarity_threshold: float = 0.85
    # Minimum cosine similarity for a Course/CourseJob to appear as a candidate
    # in the "similar courses found" preview list (create_course_job). Looser
    # than topic_similarity_threshold: the list is informational, the user
    # decides whether to navigate or force-generate, so broader/narrower
    # related courses should surface.
    topic_candidate_threshold: float = 0.70
    # Celery broker: how long a crashed worker's message stays invisible before
    # redelivery. Must comfortably exceed the longest single graph run.
    celery_visibility_timeout_seconds: int = 6 * 60 * 60
    # Hard cap on a single task run (the whole graph). A non-returning LLM call
    # cannot be rescued by tenacity, so the worker must eventually kill the
    # task; acks_late then redelivers it and the run resumes from its last
    # checkpoint instead of starting over.
    celery_task_time_limit_seconds: int = 30 * 60
    # Per-call timeout for every LLM/embedding request, so a hung provider call
    # fails (and retries/backoff) instead of blocking the graph indefinitely.
    llm_request_timeout_seconds: float = 120.0
    # Web research for course-structure generation. When disabled, the outline
    # and chapter nodes make exactly the LLM calls they made before this
    # feature existed. Search/fetch each retry once, then degrade gracefully.
    web_search_enabled: bool = True
    web_search_max_results: int = 5
    web_page_max_chars: int = 8000
    web_research_max_tool_rounds: int = 4
    web_request_timeout_seconds: float = 15.0
    # LangGraph checkpoints (partial run state) for finished jobs are pruned
    # after this many days; running jobs' checkpoints are never pruned.
    checkpoint_retention_days: int = 7
    # Exponential backoff for external LLM/embedding calls (tenacity). These are
    # workflow-scoped knobs today; they are read from Settings so a global policy
    # can reuse the same values later.
    llm_retry_max_attempts: int = 4
    llm_retry_multiplier_seconds: float = 1.0
    llm_retry_max_delay_seconds: float = 30.0
    llm_retry_jitter_seconds: float = 0.5
    # Observability. otel_enabled flips on traces, metrics and OTLP log export.
    # When it is on, the process sends OTLP to otel_exporter_otlp_endpoint
    # (host dev: http://localhost:4317; containerized: http://otel-collector:4317).
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "knovolve-backend"
    otel_log_level: str = "INFO"
    # Metrics namespace for prometheus-fastapi-instrumentator.
    metrics_namespace: str = "fastapi"
    # Sentry (self-hosted) DSN; empty string disables error tracking.
    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = 0.1
    # MinIO/S3-compatible object storage for diagram images.
    s3_endpoint: str = "http://localhost:9000"
    s3_public_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "knovolve"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    # Bound on how long the content-streaming endpoint tails Redis for a
    # still-pending diagram after all section text has been delivered, so a
    # dropped/never-publishing worker can't hang the HTTP connection forever.
    diagram_stream_timeout_seconds: float = 120.0

    class Config:
        env_file = ".env"


# pydantic-settings sources required fields (jwt_secret, opencode_api_key,
# openrouter_api_key) from environment variables / .env at runtime, but
# pyright's synthesized BaseModel __init__ has no way to know that and treats
# them as required constructor arguments.
settings = Settings()  # pyright: ignore[reportCallIssue]
