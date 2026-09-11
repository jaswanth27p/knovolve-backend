import logging
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.llm import web_research
from app.llm.web_research import read_webpage, web_search


def test_settings_defaults_present():
    s = Settings(jwt_secret="x", opencode_api_key="x", openrouter_api_key="x")
    assert s.web_search_enabled is True
    assert s.web_search_max_results == 5
    assert s.web_page_max_chars == 8000
    assert s.web_research_max_tool_rounds == 4
    assert s.web_request_timeout_seconds == 15.0


def _resp(text: str) -> MagicMock:
    m = MagicMock()
    m.text = text
    m.raise_for_status.return_value = None
    return m


def test_web_search_formats_results():
    with patch("app.llm.web_research.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.return_value = [
            {"title": "T", "href": "http://x", "body": "B"},
        ]
        out = web_search("python")
    assert "T" in out and "http://x" in out and "B" in out
    mock_ddgs.return_value.text.assert_called_once()


def test_web_search_empty_returns_no_results():
    with patch("app.llm.web_research.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.return_value = []
        assert web_search("nothing") == "No results."


def test_web_search_retries_once_then_succeeds(caplog):
    with patch("app.llm.web_research.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.side_effect = [
            RuntimeError("boom"),
            [{"title": "T", "href": "u", "body": "b"}],
        ]
        with caplog.at_level(logging.WARNING, logger="app.llm.web_research"):
            out = web_search("q")
    assert "T" in out
    assert any("retrying after error" in r.message for r in caplog.records)


def test_web_search_raises_after_one_retry(caplog):
    with patch("app.llm.web_research.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.side_effect = RuntimeError("boom")
        with caplog.at_level(logging.ERROR, logger="app.llm.web_research"):
            with pytest.raises(RuntimeError):
                web_search("q")
    assert any("failed after retry" in r.message for r in caplog.records)


def test_read_webpage_extracts_and_truncates(monkeypatch):
    monkeypatch.setattr(web_research.settings, "web_page_max_chars", 5)
    with patch("app.llm.web_research.httpx") as mock_httpx, \
         patch("app.llm.web_research.trafilatura") as mock_traf:
        mock_httpx.get.return_value = _resp("<html/>")
        mock_traf.extract.return_value = "abcdefghij"
        out = read_webpage("http://x")
    assert out == "abcde"


def test_read_webpage_no_content():
    with patch("app.llm.web_research.httpx") as mock_httpx, \
         patch("app.llm.web_research.trafilatura") as mock_traf:
        mock_httpx.get.return_value = _resp("<html/>")
        mock_traf.extract.return_value = None
        assert read_webpage("http://x") == "Could not extract content."


def test_read_webpage_retries_once(caplog):
    with patch("app.llm.web_research.httpx") as mock_httpx, \
         patch("app.llm.web_research.trafilatura") as mock_traf:
        mock_httpx.get.side_effect = [RuntimeError("boom"), _resp("<html/>")]
        mock_traf.extract.return_value = "text"
        with caplog.at_level(logging.WARNING, logger="app.llm.web_research"):
            out = read_webpage("http://x")
    assert out == "text"
    assert any("retrying after error" in r.message for r in caplog.records)
