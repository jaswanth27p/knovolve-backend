from unittest.mock import patch, MagicMock
from app.agents.assignment.nodes.generate_section_questions import QuestionDraft, SectionQuestionsResponse
from app.agents.assignment.nodes.generate_topup_questions import generate_topup_questions


def test_generate_topup_questions_requests_exact_count():
    fake_response = SectionQuestionsResponse(questions=[
        QuestionDraft(type="true_false", text="q1", options=None, correct_answer="true",
                      explanation="e", concept_tag="t1", difficulty="medium"),
    ])
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.assignment.nodes.generate_topup_questions.get_chat_model", return_value=mock_model):
        result = generate_topup_questions(
            "Chapter T", "Chapter O",
            [{"heading": "H1", "body_markdown": "body1"}, {"heading": "H2", "body_markdown": "body2"}],
            1,
        )

    assert len(result) == 1
    assert result[0].concept_tag == "t1"
    call_args = mock_model.with_structured_output.return_value.invoke.call_args
    prompt_text = str(call_args)
    assert "1" in prompt_text
