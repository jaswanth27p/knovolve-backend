from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://knovolve:knovolve@localhost:5432/knovolve"
    # SQLAlchemy connection pool. Bounded (size + overflow) so a fan-out of
    # worker threads/processes can't exhaust Postgres connections; pre_ping
    # validates a pooled connection before use (transparently replacing one
    # killed by a pg restart or network blip) and recycle_days returns idle
    # connections well under Postgres' server-side timeout.
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_recycle_seconds: int = 1800
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
    # Hard cap on a single task run for every task that does not set its own
    # time_limit/soft_time_limit (most tasks below do not — see
    # celery_course_creation_time_limit_seconds and
    # celery_generation_run_time_limit_seconds for the two that do). A
    # non-returning LLM call cannot be rescued by tenacity, so the worker must
    # eventually kill the task; acks_late then redelivers it and the run
    # resumes from its last checkpoint instead of starting over.
    celery_task_time_limit_seconds: int = 30 * 60
    # Global soft limit paired with celery_task_time_limit_seconds above.
    # Without a soft limit, Celery's hard limit sends SIGKILL directly with no
    # in-process exception at all — a task's own `except Exception` (which
    # would otherwise mark its job/row "failed") never runs, so a run that
    # overruns silently leaves its DB row stuck "running"/"pending" forever
    # instead of failing loudly. 90% of the hard cap leaves a 3-minute margin
    # at the default 30 minutes for the except block's own DB commit to land
    # before the hard SIGKILL. Keep strictly below the hard cap.
    celery_task_soft_time_limit_seconds: int = 30 * 60 * 90 // 100
    # Course creation (Learn page: outline -> chapters -> concept graph ->
    # persist) is the same class of full LLM-pipeline work as full-course
    # generation below and gets its own equally generous budget instead of the
    # generic 30-minute default, which real runs can legitimately exceed.
    # Checkpointed, so a timeout-triggered retry (see course_creation_task.py)
    # resumes rather than restarting.
    celery_course_creation_time_limit_seconds: int = 2 * 60 * 60
    # Same 95%-of-hard-cap margin rationale as celery_generation_run_soft_time_limit_seconds.
    celery_course_creation_soft_time_limit_seconds: int = 2 * 60 * 60 * 95 // 100
    # Per-call timeout for every LLM/embedding request, so a hung provider call
    # fails (and retries/backoff) instead of blocking the graph indefinitely.
    llm_request_timeout_seconds: float = 120.0
    # Web research for course-structure generation. When disabled, the outline
    # and chapter nodes make exactly the LLM calls they made before this
    # feature existed. Search/fetch each retry once, then degrade gracefully.
    web_search_enabled: bool = True
    web_search_max_results: int = 5
    web_page_max_chars: int = 8000
    # One-shot structure research, tuned independently of chapter research.
    structure_research_top_urls: int = 3
    structure_research_max_chars_per_page: int = 4000
    structure_research_max_total_chars: int = 10000
    # Course-structure nodes whose per-item work is independent (one outline
    # research + one per module) fan out up to this many threads.
    course_structure_max_workers: int = 4
    # Chapter-content web research. Governed independently of
    # web_search_enabled (which gates course-structure research) because
    # chapter content streams to a waiting learner, so its latency/cost must
    # be tunable on its own.
    #
    # Chapter research is a single deterministic pass (no agent tool loop):
    # one search, fetch the top `chapter_research_top_urls` results once, and
    # hand the extracted text to the section writers.
    chapter_research_enabled: bool = True
    chapter_research_top_urls: int = 3
    chapter_research_max_chars_per_page: int = 4000
    chapter_research_max_total_chars: int = 10000
    # Section bodies are written by independent LLM calls; run up to this many
    # concurrently so a chapter's wall-clock time is roughly
    # ceil(sections / workers) rather than sections * latency.
    chapter_section_max_workers: int = 4
    # Assignment questions are generated per teaching section; fan those LLM
    # calls out the same way as section bodies.
    assignment_question_max_workers: int = 4
    # Hard cap on sections per chapter outline. Bounds chapter wall-time; the
    # outline prompt also asks for 3-6, this is the safety net if it ignores that.
    chapter_max_sections: int = 6
    # Full-course generation: independent units (different chapters' content and
    # assignments) run up to this many at a time within one run. Combined with
    # chapter_section_max_workers this bounds concurrent LLM calls per process to
    # roughly generation_run_max_parallel_units * chapter_section_max_workers.
    generation_run_max_parallel_units: int = 4
    # Celery worker resource bounds. `concurrency` is the number of prefork
    # child processes (a hard cap on simultaneous units across ALL runs on this
    # worker). The memory/task limits recycle a child that leaks or bloats; they
    # only apply to the prefork pool, not solo.
    celery_worker_concurrency: int = 4
    celery_worker_max_memory_per_child_kb: int = 512_000
    celery_worker_max_tasks_per_child: int = 100
    # The full-course run task fans units out across threads and can legitimately
    # outlive the default 30-minute per-task cap on prefork (which solo ignored).
    # Individual units are idempotent, so a redelivered run re-does only what is
    # missing; this only needs to exceed a healthy parallel sweep.
    celery_generation_run_time_limit_seconds: int = 2 * 60 * 60
    # Soft limit must fire *before* the hard SIGKILL so the task's except block
    # can mark the run failed before the process dies; equality (the old
    # behavior) gave the except block zero margin. 95% of the hard cap leaves a
    # six-minute margin at the default 2h. Keep strictly below the hard cap.
    celery_generation_run_soft_time_limit_seconds: int = 2 * 60 * 60 * 95 // 100
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
    # Langfuse (self-hosted, docker-compose service `langfuse-web`). Mirrors
    # otel_enabled's shape: off by default, no required env vars in existing
    # dev setups.
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3400"
    langfuse_tracing_environment: str = "development"
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
        # Deliberate, and a real tradeoff. Without this, pydantic-settings
        # defaults a legacy `class Config` block to extra="forbid", and the
        # docker-compose-only secrets that share backend/.env with app config
        # (LANGFUSE_SALT, LANGFUSE_ENCRYPTION_KEY, LANGFUSE_INIT_*, the
        # per-container passwords — none of which are Settings fields) would
        # each raise a ValidationError at import time. The cost is that a
        # typo'd or renamed setting is now silently ignored project-wide
        # instead of failing loudly. The alternative — splitting .env into a
        # separate compose-only env file — was considered and rejected as more
        # invasive than the safety it buys back; see
        # docs/superpowers/specs/2026-09-18-langfuse-observability-design.md.
        extra = "ignore"


# pydantic-settings sources required fields (jwt_secret, opencode_api_key,
# openrouter_api_key) from environment variables / .env at runtime, but
# pyright's synthesized BaseModel __init__ has no way to know that and treats
# them as required constructor arguments.
settings = Settings()  # pyright: ignore[reportCallIssue]
