import json
from datetime import datetime, timezone
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter

client = TestClient(app)


def _register_and_login(email: str) -> str:
    resp = client.post("/auth/register", json={"email": email, "password": "pw123456"})
    return resp.json()["access_token"]


def _make_chapter(topic_slug: str) -> tuple[str, int]:
    with SessionLocal() as db:
        course = Course(topic_slug=topic_slug, topic_raw=topic_slug, topic_embedding=[0.0] * 2048,
                         created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
        db.add(chapter)
        db.commit()
        return course.topic_slug, chapter.id


def _events(resp) -> list[dict]:
    return [json.loads(line) for line in resp.text.strip().split("\n") if line]


def test_content_endpoint_streams_ndjson_events():
    token = _register_and_login("chapters-api-a@example.com")
    slug, chapter_id = _make_chapter("chapters-api-stream")

    fake_events = [
        {"type": "section_ready", "order": 0, "heading": "H", "kind": "teaching",
         "body_markdown": "b", "examples": [{"prompt": "p", "walkthrough": "w"}],
         "diagram_status": None, "diagram_image_url": None},
        {"type": "done"},
    ]
    with patch("app.services.chapter_content.stream_chapter_content", return_value=iter(fake_events)):
        resp = client.get(f"/courses/{slug}/chapters/{chapter_id}/content",
                           headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    events = _events(resp)
    assert events[0]["type"] == "section_ready"
    assert events[-1]["type"] == "done"


def test_content_endpoint_404_for_chapter_not_in_course():
    token = _register_and_login("chapters-api-b@example.com")
    slug, _ = _make_chapter("chapters-api-mismatch-a")
    _, other_chapter_id = _make_chapter("chapters-api-mismatch-b")

    resp = client.get(f"/courses/{slug}/chapters/{other_chapter_id}/content",
                       headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 404


def test_content_endpoint_requires_auth():
    slug, chapter_id = _make_chapter("chapters-api-noauth")
    resp = client.get(f"/courses/{slug}/chapters/{chapter_id}/content")
    assert resp.status_code in (401, 403)


def test_content_endpoint_tails_redis_for_pending_diagram():
    token = _register_and_login("chapters-api-c@example.com")
    slug, chapter_id = _make_chapter("chapters-api-diagram-tail")

    fake_events = [
        {"type": "section_ready", "order": 0, "heading": "H", "kind": "teaching",
         "body_markdown": "b", "examples": [{"prompt": "p", "walkthrough": "w"}],
         "diagram_status": "pending", "diagram_image_url": None},
        {"type": "done"},
    ]

    with SessionLocal() as db:
        from app.models.chapter_content import ChapterContent, ChapterContentSection
        now = datetime.now(timezone.utc)
        content = ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="ready",
                                  outline=[{"heading": "H", "objective": "o", "kind": "teaching"}],
                                  created_at=now, updated_at=now)
        db.add(content)
        db.commit()
        db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H", kind="teaching",
                                      body_markdown="b", examples=[{"prompt": "p", "walkthrough": "w"}],
                                      diagram_spec={"nodes": [], "edges": []},
                                      diagram_status="ready", diagram_image_url="http://x/y.svg"))
        db.commit()

    # The section's diagram already resolved to "ready" in the DB by the time
    # the endpoint reconciles pending diagrams -- it must pick that up
    # immediately rather than waiting on a redis message that will never come
    # in this test (no worker is running).
    with patch("app.services.chapter_content.stream_chapter_content", return_value=iter(fake_events)), \
         patch("app.services.chapter_content.redis.Redis.from_url") as mock_from_url:
        mock_from_url.return_value.pubsub.return_value.get_message.return_value = None
        resp = client.get(f"/courses/{slug}/chapters/{chapter_id}/content",
                           headers={"Authorization": f"Bearer {token}"})

    events = _events(resp)
    diagram_events = [e for e in events if e["type"] == "diagram_ready"]
    assert len(diagram_events) == 1
    assert diagram_events[0]["diagram_image_url"] == "http://x/y.svg"
    # Exactly one terminal event, and it comes after the diagram event.
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    assert done_events[0] is events[-1]
