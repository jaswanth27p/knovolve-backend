from app.schemas.chat import ChatRequest, ChatTurn, RouteContext


def test_chat_request_defaults_to_empty_history_and_no_context():
    req = ChatRequest(message="hi")
    assert req.history == []
    assert req.context is None
    assert req.current_route is None


def test_chat_request_accepts_history_and_route():
    req = ChatRequest(
        message="what's my progress",
        history=[ChatTurn(role="user", content="hey"), ChatTurn(role="assistant", content="hi!")],
        current_route=RouteContext(page="chapter_content", course_slug="python-basics", chapter_id=3),
    )
    assert req.history[0].role == "user"
    assert req.current_route is not None
    assert req.current_route.chapter_id == 3
