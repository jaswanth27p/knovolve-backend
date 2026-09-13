from unittest.mock import patch, MagicMock
from app.agents.course_creation.state import CourseCreationState
from app.agents.course_creation.nodes.generate_chapters import generate_chapters, ChapterDraft


def test_generate_chapters_fills_each_module():
    state: CourseCreationState = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": "typescript",
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

    modules = result["modules"]
    assert modules is not None  # generate_chapters always returns the modules list it was given
    assert len(modules[0]["chapters"]) == 1
    assert len(modules[1]["chapters"]) == 1
    assert mock_model.with_structured_output.return_value.invoke.call_count == 2


def test_generate_chapters_skips_modules_already_filled():
    """Resumability: a module with chapters already populated (from a prior
    partial run) must not be regenerated."""
    state: CourseCreationState = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": "typescript",
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

    modules = result["modules"]
    assert modules is not None  # generate_chapters always returns the modules list it was given
    assert modules[0]["chapters"][0]["title"] == "already done"  # untouched
    assert mock_model.with_structured_output.return_value.invoke.call_count == 1  # only module 2 regenerated


def test_generate_chapters_uses_web_research_per_module(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "web_search_enabled", True)

    state: CourseCreationState = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": "typescript",
              "topic_embedding": [0.0], "existing_course_id": None,
              "modules": [
                  {"title": "Basic Types", "objective": "...", "order": 1, "chapters": []},
                  {"title": "Generics", "objective": "...", "order": 2, "chapters": []},
              ],
              "concepts": None, "concept_edges": None, "error": None}

    fake_chapters = [ChapterDraft(title="Strings and Numbers", objective="...", order=1)]
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_chapters

    with patch("app.agents.course_creation.nodes.generate_chapters.get_chat_model", return_value=mock_model), \
         patch("app.agents.course_creation.nodes.generate_chapters.fetch_structure_research",
               return_value="BRIEF") as mock_research:
        generate_chapters(state)

    assert mock_research.call_count == 2  # once per module
    prompt_messages = mock_model.with_structured_output.return_value.invoke.call_args.args[0]
    assert any("BRIEF" in str(m.content) for m in prompt_messages)


def test_generate_chapters_reruns_colliding_modules_sequentially():
    """Collision reruns must be sequential: each reran module has to see its
    siblings' freshly generated titles to actually resolve the collision."""
    state: CourseCreationState = {"job_id": 1, "topic_raw": "T", "topic_slug": "t",
              "topic_embedding": [0.0], "existing_course_id": None,
              "modules": [
                  {"title": "Alpha Module", "objective": "o", "order": 1,
                   "chapters": [{"title": "Dup", "objective": "o", "order": 1}]},
                  {"title": "Beta Module", "objective": "o", "order": 2,
                   "chapters": [{"title": "Dup", "objective": "o", "order": 1}]},
              ],
              "concepts": None, "concept_edges": None, "error": None,
              "rerun_modules": ["Alpha Module", "Beta Module"]}

    beta_prompts: list[str] = []

    def fake_invoke(messages):
        text = "\n".join(str(m.content) for m in messages)
        if "Alpha Module" in text:
            return [ChapterDraft(title="Alpha Chapter", objective="o", order=1)]
        beta_prompts.append(text)
        return [ChapterDraft(title="Beta Chapter", objective="o", order=1)]

    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.side_effect = fake_invoke

    with patch("app.agents.course_creation.nodes.generate_chapters.get_chat_model", return_value=mock_model):
        result = generate_chapters(state)

    modules = result["modules"]
    assert modules is not None
    assert modules[0]["chapters"][0]["title"] == "Alpha Chapter"
    assert modules[1]["chapters"][0]["title"] == "Beta Chapter"
    # Beta was generated after Alpha and saw Alpha's fresh title in its prompt.
    assert beta_prompts and any("Alpha Chapter" in text for text in beta_prompts)
