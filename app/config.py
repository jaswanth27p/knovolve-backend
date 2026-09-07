from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://knovolve:knovolve@localhost:5432/knovolve"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str
    jwt_access_ttl_minutes: int = 15
    jwt_refresh_ttl_days: int = 30
    opencode_api_key: str
    opencode_zen_api_key: str = ""
    openrouter_api_key: str
    # JSON list in .env, e.g. CORS_ORIGINS=["http://localhost:3000"]
    cors_origins: list[str] = ["http://localhost:3000"]
    # Minimum cosine similarity for a Course/CourseJob to count as an existing
    # version of a requested topic (dedup).
    topic_similarity_threshold: float = 0.85
    # Celery broker: how long a crashed worker's message stays invisible before
    # redelivery. Must comfortably exceed the longest single graph run.
    celery_visibility_timeout_seconds: int = 6 * 60 * 60
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

    class Config:
        env_file = ".env"


# pydantic-settings sources required fields (jwt_secret, opencode_api_key,
# openrouter_api_key) from environment variables / .env at runtime, but
# pyright's synthesized BaseModel __init__ has no way to know that and treats
# them as required constructor arguments.
settings = Settings()  # pyright: ignore[reportCallIssue]
