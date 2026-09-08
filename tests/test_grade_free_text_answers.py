from unittest.mock import patch, MagicMock
from app.agents.evaluation.nodes.grade_free_text_answers import (
    grade_free_text_answers,
    FreeTextAnswerItem,
    AnswerGrade,
    GradingResponse,
)


def test_grade_free_text_answers_returns_one_grade_per_item():
    items = [
        FreeTextAnswerItem(question_id=1, question_text="What is X?", correct_answer="X is Y",
                            explanation="e1", user_answer="X is Y basically"),
        FreeTextAnswerItem(question_id=2, question_text="What is Z?", correct_answer="Z is W",
                            explanation="e2", user_answer="no idea"),
    ]
    fake_response = GradingResponse(grades=[
        AnswerGrade(question_id=1, is_correct=True, feedback="Correct, that matches the definition."),
        AnswerGrade(question_id=2, is_correct=False, feedback="Incorrect — Z is W, not left blank."),
    ])
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.evaluation.nodes.grade_free_text_answers.get_chat_model", return_value=mock_model):
        result = grade_free_text_answers(items)

    assert len(result) == 2
    assert result[0].question_id == 1
    assert result[0].is_correct is True
    assert result[1].is_correct is False

    # Batched: exactly one invoke() call regardless of item count.
    mock_model.with_structured_output.return_value.invoke.assert_called_once()
    prompt_text = str(mock_model.with_structured_output.return_value.invoke.call_args)
    assert "What is X?" in prompt_text
    assert "What is Z?" in prompt_text
