from unittest.mock import patch, MagicMock
from app.agents.chapter_content.nodes.generate_section_outline import (
    generate_section_outline,
    SectionOutlineDraft,
)


def test_generate_section_outline_returns_sorted_sections():
    fake_sections = [
        SectionOutlineDraft(heading="Applying It", objective="...", kind="teaching", order=2),
        SectionOutlineDraft(heading="Why This Matters", objective="...", kind="intro", order=1),
    ]
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_sections

    with patch("app.agents.chapter_content.nodes.generate_section_outline.get_chat_model", return_value=mock_model):
        result = generate_section_outline("Generic Functions", "Write reusable typed functions")

    assert [s.heading for s in result] == ["Why This Matters", "Applying It"]
    assert result[0].kind == "intro"
    assert result[1].kind == "teaching"
