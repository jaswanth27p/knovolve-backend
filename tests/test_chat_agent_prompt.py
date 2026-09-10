from app.llm.prompts import CHAT_AGENT_SYSTEM_PROMPT


def test_prompt_embeds_bundle_and_route_and_mentions_markdown():
    messages = CHAT_AGENT_SYSTEM_PROMPT.format_messages(
        bundle_json='{"in_progress_count": 1}', route_json='{"page": "dashboard"}',
    )
    system_content = messages[0].content
    assert isinstance(system_content, str)
    assert '"in_progress_count": 1' in system_content
    assert '"page": "dashboard"' in system_content
    assert "markdown" in system_content.lower()
