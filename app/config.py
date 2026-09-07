from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://knovolve:knovolve@localhost:5432/knovolve"
    redis_url: str = "redis://localhost:6379/0"
    jwt_secret: str
    jwt_access_ttl_minutes: int = 15
    jwt_refresh_ttl_days: int = 30
    opencode_api_key: str
    openrouter_api_key: str

    class Config:
        env_file = ".env"


# pydantic-settings sources required fields (jwt_secret, opencode_api_key,
# openrouter_api_key) from environment variables / .env at runtime, but
# pyright's synthesized BaseModel __init__ has no way to know that and treats
# them as required constructor arguments.
settings = Settings()  # pyright: ignore[reportCallIssue]
