from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.chat import ChatResponse

client = TestClient(app)


def _auth_headers():
    client.post("/auth/register", json={"email": "me-chat@example.com", "password": "pw123456"})
    resp = client.post("/auth/login", json={"email": "me-chat@example.com", "password": "pw123456"})
    return {"Authorization": f"Bearer {resp.cookies['access_token']}"}


def test_post_chat_requires_auth():
    resp = client.post("/me/chat", json={"message": "hi"})
    assert resp.status_code == 401


def test_post_chat_passes_full_request_through_to_service():
    headers = _auth_headers()
    with patch(
        "app.services.chat.answer_chat_message",
        return_value=ChatResponse(reply="hello!", context=None),
    ) as mock_answer:
        resp = client.post(
            "/me/chat",
            json={
                "message": "hi",
                "history": [{"role": "user", "content": "earlier"}],
                "current_route": {"page": "dashboard"},
            },
            headers=headers,
        )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "hello!"
    called_req = mock_answer.call_args.kwargs.get("req") or mock_answer.call_args.args[-1]
    assert called_req.message == "hi"
    assert called_req.history[0].content == "earlier"
    assert called_req.current_route.page == "dashboard"


def test_post_chat_stream_streams_ndjson_events():
    headers = _auth_headers()
    with patch(
        "app.services.chat.stream_chat_message",
        return_value=iter([{"type": "token", "text": "hi"}, {"type": "done"}]),
    ):
        resp = client.post("/me/chat/stream", json={"message": "hi"}, headers=headers)
    lines = [line for line in resp.text.strip().split("\n") if line]
    assert '"type": "token"' in lines[0] or '"type":"token"' in lines[0]
