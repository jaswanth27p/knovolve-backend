import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError
from app.agents.assignment.nodes.generate_section_questions import (
    generate_questions_for_section,
    QuestionDraft,
    SectionQuestionsResponse,
)


def test_generate_questions_for_section_returns_one_per_subtopic():
    fake_response = SectionQuestionsResponse(questions=[
        QuestionDraft(type="mcq", text="What is X?", options=["a", "b"], correct_answer="a",
                      explanation="e", concept_tag="X definition", difficulty="easy"),
        QuestionDraft(type="free_text", text="Explain Y", options=None, correct_answer="model answer",
                      explanation="e2", concept_tag="Y gotcha", difficulty="hard"),
    ])
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.assignment.nodes.generate_section_questions.get_chat_model", return_value=mock_model):
        result = generate_questions_for_section(
            "Chapter T", "Chapter O", "Section H", "body text", [{"prompt": "p", "walkthrough": "w"}],
        )

    assert len(result) == 2
    assert result[0].concept_tag == "X definition"
    assert result[1].type == "free_text"


def test_mcq_question_requires_at_least_two_options():
    with pytest.raises(ValidationError):
        QuestionDraft(type="mcq", text="q", options=["only-one"], correct_answer="only-one",
                      explanation="e", concept_tag="t", difficulty="easy")


def test_mcq_question_requires_options_not_none():
    with pytest.raises(ValidationError):
        QuestionDraft(type="mcq", text="q", options=None, correct_answer="a",
                      explanation="e", concept_tag="t", difficulty="easy")
