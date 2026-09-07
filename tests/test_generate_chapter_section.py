from unittest.mock import patch, MagicMock
import pytest
from app.agents.chapter_content.nodes.generate_chapter_section import (
    generate_chapter_section,
    ChapterSectionResponse,
    ExampleDraft,
    DiagramSpecDraft,
    DiagramNodeDraft,
    DiagramEdgeDraft,
)


def test_teaching_section_with_examples_returned_as_is():
    fake = ChapterSectionResponse(
        body_markdown="Some content.",
        examples=[ExampleDraft(prompt="p", walkthrough="w")],
        diagram_spec=None,
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake

    with patch("app.agents.chapter_content.nodes.generate_chapter_section.get_chat_model", return_value=mock_model):
        result = generate_chapter_section("Ch", "obj", "Heading", "sec obj", "teaching")

    assert result.examples[0].prompt == "p"
    assert mock_model.with_structured_output.return_value.invoke.call_count == 1


def test_teaching_section_with_no_examples_triggers_corrective_retry():
    empty = ChapterSectionResponse(body_markdown="Some content.", examples=[], diagram_spec=None)
    fixed = ChapterSectionResponse(
        body_markdown="Some content.",
        examples=[ExampleDraft(prompt="p", walkthrough="w")],
        diagram_spec=None,
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.side_effect = [empty, fixed]

    with patch("app.agents.chapter_content.nodes.generate_chapter_section.get_chat_model", return_value=mock_model):
        result = generate_chapter_section("Ch", "obj", "Heading", "sec obj", "teaching")

    assert len(result.examples) == 1
    assert mock_model.with_structured_output.return_value.invoke.call_count == 2


def test_teaching_section_still_empty_after_retry_raises():
    empty = ChapterSectionResponse(body_markdown="Some content.", examples=[], diagram_spec=None)
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.side_effect = [empty, empty]

    with patch("app.agents.chapter_content.nodes.generate_chapter_section.get_chat_model", return_value=mock_model):
        with pytest.raises(ValueError, match="zero examples"):
            generate_chapter_section("Ch", "obj", "Heading", "sec obj", "teaching")


def test_intro_section_allows_empty_examples_no_retry():
    fake = ChapterSectionResponse(body_markdown="Welcome.", examples=[], diagram_spec=None)
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake

    with patch("app.agents.chapter_content.nodes.generate_chapter_section.get_chat_model", return_value=mock_model):
        result = generate_chapter_section("Ch", "obj", "Intro", "sec obj", "intro")

    assert result.examples == []
    assert mock_model.with_structured_output.return_value.invoke.call_count == 1


def test_diagram_spec_round_trips():
    fake = ChapterSectionResponse(
        body_markdown="Content.",
        examples=[ExampleDraft(prompt="p", walkthrough="w")],
        diagram_spec=DiagramSpecDraft(
            nodes=[DiagramNodeDraft(id="a", label="A"), DiagramNodeDraft(id="b", label="B")],
            edges=[DiagramEdgeDraft(source="a", target="b", label=None)],
        ),
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake

    with patch("app.agents.chapter_content.nodes.generate_chapter_section.get_chat_model", return_value=mock_model):
        result = generate_chapter_section("Ch", "obj", "Heading", "sec obj", "teaching")

    assert result.diagram_spec is not None
    assert result.diagram_spec.edges[0].source == "a"
