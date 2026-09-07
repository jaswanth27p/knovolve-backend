import os
import pytest
from unittest.mock import patch, MagicMock
from app.llm.factory import get_chat_model, embed
from app.llm import config as llm_config

def test_get_chat_model_uses_registered_provider(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    model = get_chat_model("generate_outline")
    assert model.openai_api_base == "https://opencode.ai/zen/go/v1"
    assert model.model_name == "mimo-v2.5"

def test_get_chat_model_unknown_node_raises():
    with pytest.raises(KeyError):
        get_chat_model("not_a_real_node")

def test_embed_calls_configured_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("app.llm.factory._embeddings_client") as mock_client:
        mock_client.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
        result = embed("closures in javascript")
        assert result == [0.1, 0.2, 0.3]

def test_swapping_provider_is_config_only(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "opencode_zen_api_key", "test-key")
    original = llm_config.LLM_NODES["generate_outline"]
    llm_config.LLM_NODES["generate_outline"] = {"provider": "opencode-zen", "model": "gpt-5.6"}
    try:
        model = get_chat_model("generate_outline")
        assert model.openai_api_base == "https://opencode.ai/zen/v1"
    finally:
        llm_config.LLM_NODES["generate_outline"] = original
