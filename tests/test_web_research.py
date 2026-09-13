import logging
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.llm import web_research
from app.llm.web_research import (
    read_webpage_once,
    web_search,
)


def test_settings_defaults_present():
    s = Settings(jwt_secret="x", opencode_api_key="x", openrouter_api_key="x")
    assert s.web_search_enabled is True
    assert s.web_search_max_results == 5
    assert s.web_page_max_chars == 8000
    assert s.web_request_timeout_seconds == 15.0
    assert s.chapter_research_enabled is True
    assert s.chapter_research_top_urls == 3
    assert s.chapter_research_max_chars_per_page == 4000
    assert s.chapter_research_max_total_chars == 10000
    assert s.structure_research_top_urls == 3
    assert s.structure_research_max_chars_per_page == 4000
    assert s.structure_research_max_total_chars == 10000
    assert s.course_structure_max_workers == 4
    assert s.chapter_max_sections == 6
    assert s.generation_run_max_parallel_units == 4
    assert s.assignment_question_max_workers == 4
    assert s.celery_worker_concurrency == 4
    assert s.celery_worker_max_memory_per_child_kb == 512_000
    assert s.celery_worker_max_tasks_per_child == 100
    assert s.celery_generation_run_time_limit_seconds == 7200


def test_fetch_structure_research_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(web_research.settings, "web_search_enabled", False)
    with patch("app.llm.web_research.web_search") as mock_search:
        assert web_research.fetch_structure_research("topic") == ""
    mock_search.assert_not_called()


def test_fetch_structure_research_one_shot(monkeypatch):
    monkeypatch.setattr(web_research.settings, "web_search_enabled", True)
    monkeypatch.setattr(web_research.settings, "structure_research_top_urls", 1)
    with patch("app.llm.web_research.web_search", return_value="T | http://a | s"), \
         patch("app.llm.web_research.read_webpage_once", return_value="body"):
        out = web_research.fetch_structure_research("topic")
    assert "URL: http://a" in out and "body" in out


def test_fetch_chapter_research_one_search_top_urls_no_retry():
    with patch("app.llm.web_research.web_search", return_value=(
        "T1 | http://a | s\nT2 | http://b | s\nT3 | http://c | s"
    )) as mock_search, \
         patch("app.llm.web_research.read_webpage_once", side_effect=lambda u: f"body-{u}") as mock_fetch:
        out = web_research.fetch_chapter_research(
            "q", top_urls=2, max_chars_per_page=100, max_total_chars=1000
        )
    mock_search.assert_called_once_with("q")
    assert mock_fetch.call_count == 2
    assert "URL: http://a" in out and "body-http://a" in out
    assert "URL: http://b" in out
    assert "http://c" not in out


def test_fetch_chapter_research_empty_when_no_results():
    with patch("app.llm.web_research.web_search", return_value="No results."):
        assert web_research.fetch_chapter_research("q") == ""


def test_read_webpage_once_does_not_retry():
    with patch("app.llm.web_research._fetch_and_extract", side_effect=RuntimeError("boom")) as mock_fetch:
        assert web_research.read_webpage_once("http://x") == "Could not extract content."
    assert mock_fetch.call_count == 1


def _resp(text: str) -> MagicMock:
    m = MagicMock()
    m.text = text
    m.status_code = 200
    m.is_redirect = False
    m.headers = {}
    m.raise_for_status.return_value = None
    return m


def _redirect(location: str) -> MagicMock:
    m = MagicMock()
    m.status_code = 302
    m.is_redirect = True
    m.headers = {"location": location}
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


def test_web_search_pins_backends():
    with patch("app.llm.web_research.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.return_value = [
            {"title": "T", "href": "http://x", "body": "B"},
        ]
        web_search("python")
    _, kwargs = mock_ddgs.return_value.text.call_args
    assert "google" not in kwargs["backend"].split(",")
    assert "brave" not in kwargs["backend"].split(",")


def test_web_search_failure_degrades_to_no_results(caplog):
    with patch("app.llm.web_research.DDGS") as mock_ddgs:
        mock_ddgs.return_value.text.side_effect = RuntimeError("boom")
        with caplog.at_level(logging.WARNING, logger="app.llm.web_research"):
            assert web_search("q") == "No results."
    mock_ddgs.return_value.text.assert_called_once()
    assert any("failed fast" in r.message for r in caplog.records)


def test_fetch_and_extract_extracts_and_truncates(monkeypatch):
    monkeypatch.setattr(web_research.settings, "web_page_max_chars", 5)
    with patch("app.llm.web_research.httpx") as mock_httpx, \
         patch("app.llm.web_research.trafilatura") as mock_traf:
        mock_httpx.get.return_value = _resp("<html/>")
        mock_traf.extract.return_value = "abcdefghij"
        out = web_research._fetch_and_extract("http://93.184.216.34/")
    assert out == "abcde"


def test_read_webpage_once_no_content():
    with patch("app.llm.web_research.httpx") as mock_httpx, \
         patch("app.llm.web_research.trafilatura") as mock_traf:
        mock_httpx.get.return_value = _resp("<html/>")
        mock_traf.extract.return_value = None
        assert read_webpage_once("http://93.184.216.34/") == "Could not extract content."


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/",
        "http://10.0.0.5/",
        "file:///etc/passwd",
    ],
)
def test_fetch_and_extract_rejects_non_public_urls(url):
    with patch("app.llm.web_research.httpx") as mock_httpx:
        with pytest.raises(ValueError):
            web_research._fetch_and_extract(url)
    mock_httpx.get.assert_not_called()


def test_fetch_and_extract_rejects_private_redirect():
    with patch("app.llm.web_research.httpx") as mock_httpx:
        mock_httpx.get.return_value = _redirect("http://127.0.0.1/secret")
        with pytest.raises(ValueError):
            web_research._fetch_and_extract("http://93.184.216.34/")
    fetched = [c.args[0] for c in mock_httpx.get.call_args_list]
    assert all("127.0.0.1" not in u for u in fetched)
