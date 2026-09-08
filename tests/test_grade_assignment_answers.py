from unittest.mock import patch, MagicMock
from app.agents.evaluation.nodes.grade_assignment_answers import (
    grade_assignment_answers,
    FreeTextAnswerItem,
    KnownAnswerItem,
    AnswerGrade,
    GradingResponse,
)


def test_grades_each_free_text_item_and_returns_holistic_verdict():
    free_text_items = [
        FreeTextAnswerItem(question_id=1, question_text="What is X?", correct_answer="X is Y",
                            explanation="e1", concept_tag="t1", user_answer="X is Y basically"),
        FreeTextAnswerItem(question_id=2, question_text="What is Z?", correct_answer="Z is W",
                            explanation="e2", concept_tag="t2", user_answer="no idea"),
    ]
    fake_response = GradingResponse(
        grades=[
            AnswerGrade(question_id=1, is_correct=True, feedback="Correct, matches the definition."),
            AnswerGrade(question_id=2, is_correct=False, feedback="Incorrect.", misconception_tag="left-blank"),
        ],
        remediation_concept_tags=["t2"],
        verdict_reasoning="Understood t1 well; missed t2 entirely.",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.evaluation.nodes.grade_assignment_answers.get_chat_model", return_value=mock_model):
        result = grade_assignment_answers(free_text_items, [])

    assert len(result.grades) == 2
    assert result.grades[0].question_id == 1
    assert result.grades[0].is_correct is True
    assert result.grades[0].misconception_tag is None
    assert result.grades[1].is_correct is False
    assert result.grades[1].misconception_tag == "left-blank"
    assert result.remediation_concept_tags == ["t2"]
    assert result.verdict_reasoning == "Understood t1 well; missed t2 entirely."

    # Batched: exactly one invoke() call regardless of item count.
    mock_model.with_structured_output.return_value.invoke.assert_called_once()
    prompt_text = str(mock_model.with_structured_output.return_value.invoke.call_args)
    assert "What is X?" in prompt_text
    assert "What is Z?" in prompt_text


def test_runs_with_zero_free_text_items_for_verdict_only():
    known_answers = [
        KnownAnswerItem(question_id=10, question_text="Pick one", concept_tag="t1",
                         user_answer="a", is_correct=True),
        KnownAnswerItem(question_id=11, question_text="True or false", concept_tag="t2",
                         user_answer="false", is_correct=False),
    ]
    fake_response = GradingResponse(
        grades=[], remediation_concept_tags=["t2"], verdict_reasoning="Missed t2 on the true/false question.",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.evaluation.nodes.grade_assignment_answers.get_chat_model", return_value=mock_model):
        result = grade_assignment_answers([], known_answers)

    assert result.grades == []
    assert result.remediation_concept_tags == ["t2"]
    mock_model.with_structured_output.return_value.invoke.assert_called_once()
    prompt_text = str(mock_model.with_structured_output.return_value.invoke.call_args)
    assert "Pick one" in prompt_text
    assert "True or false" in prompt_text


def test_known_answers_included_as_context_not_regraded():
    free_text_items = [
        FreeTextAnswerItem(question_id=1, question_text="What is X?", correct_answer="X is Y",
                            explanation="e1", concept_tag="t1", user_answer="X is Y"),
    ]
    known_answers = [
        KnownAnswerItem(question_id=2, question_text="Pick one", concept_tag="t2",
                         user_answer="a", is_correct=True),
    ]
    # Mock returns a grade for both the free-text question (id 1) AND a hallucinated
    # grade for the known-answer question (id 2) — the SUT must filter out the hallucination.
    fake_response = GradingResponse(
        grades=[
            AnswerGrade(question_id=1, is_correct=True, feedback="Correct."),
            AnswerGrade(question_id=2, is_correct=True, feedback="Hallucinated grade for known answer."),
        ],
        remediation_concept_tags=[], verdict_reasoning="All good.",
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.evaluation.nodes.grade_assignment_answers.get_chat_model", return_value=mock_model):
        result = grade_assignment_answers(free_text_items, known_answers)

    # The filter must strip out the hallucinated grade for question_id 2 (the known answer).
    # Only the free-text question (id 1) should remain.
    assert [g.question_id for g in result.grades] == [1]
    assert len(result.grades) == 1
    assert result.grades[0].question_id == 1
    prompt_text = str(mock_model.with_structured_output.return_value.invoke.call_args)
    assert "Pick one" in prompt_text  # known answer passed as context only
