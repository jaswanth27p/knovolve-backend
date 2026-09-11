from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _auth_headers():
    client.post("/auth/register", json={"email": "custom-clarify@example.com", "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": "custom-clarify@example.com", "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}


def test_clarify_requires_auth():
    assert client.post("/courses/anything/export-clarify", json={"message": "hi"}).status_code == 401


def test_clarify_returns_plan_response():
    from app.db import SessionLocal
    from app.models.course import Course
    from datetime import datetime, timezone
    headers = _auth_headers()
    with SessionLocal() as db:
        db.add(Course(topic_slug="custom-clarify-course", topic_raw="Clarify",
                      topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc)))
        db.commit()
    plan = {
        "title": "Interview preparation",
        "output_kind": "qa",
        "length": "medium",
        "item_count": 3,
        "notes": "Questions first, then answers.",
    }
    agent_result = {"type": "plan", "reply": "Ready.", "questions": [], "plan": plan}
    with patch("app.services.exports.run_clarify", return_value=agent_result):
        resp = client.post("/courses/custom-clarify-course/export-clarify",
                           json={"message": "Interview questions", "history": []}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["plan"]["item_count"] == 3
