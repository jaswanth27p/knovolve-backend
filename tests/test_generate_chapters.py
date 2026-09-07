from unittest.mock import patch, MagicMock
from app.agents.course_creation.nodes.generate_chapters import generate_chapters, ChapterDraft


def test_generate_chapters_fills_each_module():
    state = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": "typescript",
              "topic_embedding": [0.0], "existing_course_id": None,
              "modules": [
                  {"title": "Basic Types", "objective": "...", "order": 1, "chapters": []},
                  {"title": "Generics", "objective": "...", "order": 2, "chapters": []},
              ],
              "concepts": None, "concept_edges": None, "error": None}

    fake_chapters = [ChapterDraft(title="Strings and Numbers", objective="...", order=1)]
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_chapters

    with patch("app.agents.course_creation.nodes.generate_chapters.get_chat_model", return_value=mock_model):
        result = generate_chapters(state)

    assert len(result["modules"][0]["chapters"]) == 1
    assert len(result["modules"][1]["chapters"]) == 1
    assert mock_model.with_structured_output.return_value.invoke.call_count == 2


def test_generate_chapters_skips_modules_already_filled():
    """Resumability: a module with chapters already populated (from a prior
    partial run) must not be regenerated."""
    state = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": "typescript",
              "topic_embedding": [0.0], "existing_course_id": None,
              "modules": [
                  {"title": "Basic Types", "objective": "...", "order": 1,
                   "chapters": [{"title": "already done", "objective": "x", "order": 1}]},
                  {"title": "Generics", "objective": "...", "order": 2, "chapters": []},
              ],
              "concepts": None, "concept_edges": None, "error": None}

    fake_chapters = [ChapterDraft(title="Generic Functions", objective="...", order=1)]
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_chapters

    with patch("app.agents.course_creation.nodes.generate_chapters.get_chat_model", return_value=mock_model):
        result = generate_chapters(state)

    assert result["modules"][0]["chapters"][0]["title"] == "already done"  # untouched
    assert mock_model.with_structured_output.return_value.invoke.call_count == 1  # only module 2 regenerated
