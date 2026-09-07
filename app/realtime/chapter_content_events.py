import json
import redis
from app.config import settings


def channel_name(chapter_content_id: int) -> str:
    return f"chapter_content:{chapter_content_id}"


def publish_event(chapter_content_id: int, event: dict) -> None:
    client = redis.Redis.from_url(settings.redis_url)
    client.publish(channel_name(chapter_content_id), json.dumps(event))
