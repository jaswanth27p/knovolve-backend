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


settings = Settings()
