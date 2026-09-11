import logging
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.config import Settings
from app.llm import web_research
from app.llm.web_research import (
    gather_research,
    read_webpage,
    run_web_research,
    web_search,
)


def test_settings_defaults_present():
    s = Settings(jwt_secret="x", opencode_api_key="x", openrouter_api_key="x")
    assert s.web_search_enabled is True
    assert s.web_search_max_results == 5
    assert s.web_page_max_chars == 8000
    assert s.web_research_max_tool_rounds == 4
    assert s.web_research_max_tool_calls == 8
    assert s.web_request_timeout_seconds == 15.0
    assert s.chapter_research_enabled is True
    assert s.chapter_research_max_tool_rounds == 3
    assert s.chapter_research_max_tool_calls == 6


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
        out = read_webpage("http://93.184.216.34/")
    assert out == "abcde"


def test_read_webpage_no_content():
    with patch("app.llm.web_research.httpx") as mock_httpx, \
         patch("app.llm.web_research.trafilatura") as mock_traf:
        mock_httpx.get.return_value = _resp("<html/>")
        mock_traf.extract.return_value = None
        assert read_webpage("http://93.184.216.34/") == "Could not extract content."


def test_read_webpage_retries_once(caplog):
    with patch("app.llm.web_research.httpx") as mock_httpx, \
         patch("app.llm.web_research.trafilatura") as mock_traf:
        mock_httpx.get.side_effect = [RuntimeError("boom"), _resp("<html/>")]
        mock_traf.extract.return_value = "text"
        with caplog.at_level(logging.WARNING, logger="app.llm.web_research"):
            out = read_webpage("http://93.184.216.34/")
    assert out == "text"
    assert any("retrying after error" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/",
        "http://10.0.0.5/",
        "file:///etc/passwd",
    ],
)
def test_read_webpage_rejects_non_public_urls(url):
    with patch("app.llm.web_research.httpx") as mock_httpx:
        with pytest.raises(ValueError):
            read_webpage(url)
    mock_httpx.get.assert_not_called()


def test_read_webpage_rejects_private_redirect():
    with patch("app.llm.web_research.httpx") as mock_httpx:
        mock_httpx.get.return_value = _redirect("http://127.0.0.1/secret")
        with pytest.raises(ValueError):
            read_webpage("http://93.184.216.34/")
    fetched = [c.args[0] for c in mock_httpx.get.call_args_list]
    assert all("127.0.0.1" not in u for u in fetched)


def _tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def test_gather_research_no_tool_calls_returns_content(caplog):
    model = MagicMock()
    model.invoke.return_value = AIMessage(content="notes")
    with caplog.at_level(logging.INFO, logger="app.llm.web_research"):
        out = gather_research(model, [], 4)
    assert out == "notes"
    assert any("web research finished" in r.message for r in caplog.records)


def test_gather_research_executes_tool_then_returns():
    model = MagicMock()
    model.invoke.side_effect = [
        _tool_call("web_search", {"query": "q"}),
        AIMessage(content="final notes"),
    ]
    with patch("app.llm.web_research.web_search", return_value="RESULT") as mock_search:
        out = gather_research(model, [], 4)
    assert out == "final notes"
    mock_search.assert_called_once_with(query="q")


def test_gather_research_tool_error_degrades():
    model = MagicMock()
    model.invoke.side_effect = [
        _tool_call("web_search", {"query": "q"}),
        AIMessage(content="notes after error"),
    ]
    with patch("app.llm.web_research.web_search", side_effect=RuntimeError("ddg down")):
        out = gather_research(model, [], 4)
    assert out == "notes after error"


def test_gather_research_unknown_tool_degrades():
    model = MagicMock()
    model.invoke.side_effect = [
        _tool_call("nope", {}),
        AIMessage(content="ok"),
    ]
    out = gather_research(model, [], 4)
    assert out == "ok"


def test_gather_research_budget_exhausted_returns_empty(caplog):
    model = MagicMock()
    model.invoke.return_value = _tool_call("web_search", {"query": "q"})
    with patch("app.llm.web_research.web_search", return_value="R"):
        with caplog.at_level(logging.WARNING, logger="app.llm.web_research"):
            out = gather_research(model, [], 2)
    assert out == ""
    assert any("yielded no context" in r.message for r in caplog.records)


def test_gather_research_tool_budget_caps_executions(caplog):
    model = MagicMock()
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "a"}, "id": "c1", "type": "tool_call"},
                {"name": "web_search", "args": {"query": "b"}, "id": "c2", "type": "tool_call"},
            ],
        ),
        AIMessage(content="final notes"),
    ]
    with patch("app.llm.web_research.web_search", return_value="R") as mock_search:
        with caplog.at_level(logging.WARNING, logger="app.llm.web_research"):
            out = gather_research(model, [], 4, max_tool_calls=1)
    assert out == ""
    assert mock_search.call_count == 1
    assert any("tool budget exhausted" in r.message for r in caplog.records)


def test_gather_research_summarizes_when_round_budget_exhausted():
    model = MagicMock()
    model.invoke.return_value = _tool_call("web_search", {"query": "q"})
    summarizer = MagicMock()
    summarizer.invoke.return_value = AIMessage(content="SUMMARY NOTES")
    with patch("app.llm.web_research.web_search", return_value="R"):
        out = gather_research(model, [], 2, summarizer=summarizer)
    assert out == "SUMMARY NOTES"
    summarizer.invoke.assert_called_once()


def test_gather_research_summarizes_when_tool_cap_trips():
    model = MagicMock()
    model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {"query": "a"}, "id": "c1", "type": "tool_call"},
            {"name": "web_search", "args": {"query": "b"}, "id": "c2", "type": "tool_call"},
        ],
    )
    summarizer = MagicMock()
    summarizer.invoke.return_value = AIMessage(content="SUMMARY NOTES")
    with patch("app.llm.web_research.web_search", return_value="R"):
        out = gather_research(
            model, [], 3, max_tool_calls=1, summarizer=summarizer
        )
    assert out == "SUMMARY NOTES"


def test_gather_research_no_tools_no_summary_when_no_results():
    model = MagicMock()
    model.invoke.return_value = AIMessage(content="")
    summarizer = MagicMock()
    out = gather_research(model, [], 2, summarizer=summarizer)
    assert out == ""
    summarizer.invoke.assert_not_called()


def test_run_web_research_disabled(monkeypatch):
    monkeypatch.setattr(web_research.settings, "web_search_enabled", False)
    model = MagicMock()
    assert run_web_research(model, []) == ""
    model.bind_tools.assert_not_called()


def test_run_web_research_enabled_binds_tools(monkeypatch):
    monkeypatch.setattr(web_research.settings, "web_search_enabled", True)
    model = MagicMock()
    model.bind_tools.return_value.invoke.return_value = AIMessage(content="brief")
    out = run_web_research(model, [])
    assert out == "brief"
    model.bind_tools.assert_called_once()


def test_run_web_research_degrades_on_model_failure(monkeypatch, caplog):
    monkeypatch.setattr(web_research.settings, "web_search_enabled", True)
    model = MagicMock()
    with patch(
        "app.llm.web_research.gather_research", side_effect=RuntimeError("bind boom")
    ):
        with caplog.at_level(logging.WARNING, logger="app.llm.web_research"):
            out = run_web_research(model, [])
    assert out == ""
    assert any("web research unavailable" in r.message for r in caplog.records)
