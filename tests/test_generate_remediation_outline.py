from unittest.mock import patch, MagicMock
from app.agents.chapter_content.nodes.generate_section_outline import SectionOutlineDraft, SectionOutlineResponse
from app.agents.chapter_content.nodes.generate_remediation_outline import generate_remediation_outline


def test_generates_narrow_outline_from_weak_concepts():
    fake_response = SectionOutlineResponse(sections=[
        SectionOutlineDraft(heading="Recursion base cases", objective="Identify a base case", kind="teaching", order=1),
        SectionOutlineDraft(heading="Stack frames on recursive calls", objective="Trace a call stack", kind="teaching", order=2),
    ])
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.chapter_content.nodes.generate_remediation_outline.get_chat_model", return_value=mock_model):
        result = generate_remediation_outline("Recursion", "Understand recursive functions", ["recursion-base-case"])

    assert len(result) == 2
    assert result[0].order == 1
    assert result[1].order == 2
    prompt_text = str(mock_model.with_structured_output.return_value.invoke.call_args)
    assert "recursion-base-case" in prompt_text


def test_truncates_to_at_most_three_sections():
    fake_response = SectionOutlineResponse(sections=[
        SectionOutlineDraft(heading=f"H{i}", objective="o", kind="teaching", order=i) for i in range(1, 6)
    ])
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.chapter_content.nodes.generate_remediation_outline.get_chat_model", return_value=mock_model):
        result = generate_remediation_outline("Recursion", "Understand recursive functions", ["t1", "t2"])

    assert len(result) == 3
    assert [s.order for s in result] == [1, 2, 3]


def test_raises_on_zero_sections():
    fake_response = SectionOutlineResponse(sections=[])
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.chapter_content.nodes.generate_remediation_outline.get_chat_model", return_value=mock_model):
        try:
            generate_remediation_outline("Recursion", "Understand recursive functions", ["t1"])
            assert False, "expected ValueError"
        except ValueError:
            pass
