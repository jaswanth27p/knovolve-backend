import json
import redis
from app.config import settings

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.redis_url)
    return _client


def channel_name(chapter_content_id: int) -> str:
    return f"chapter_content:{chapter_content_id}"


def publish_event(chapter_content_id: int, event: dict) -> None:
    _get_client().publish(channel_name(chapter_content_id), json.dumps(event))