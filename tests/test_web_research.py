from app.config import Settings


def test_settings_defaults_present():
    s = Settings(jwt_secret="x", opencode_api_key="x", openrouter_api_key="x")
    assert s.web_search_enabled is True
    assert s.web_search_max_results == 5
    assert s.web_page_max_chars == 8000
    assert s.web_research_max_tool_rounds == 4
    assert s.web_request_timeout_seconds == 15.0
