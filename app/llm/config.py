LLM_NODES = {
    "canonicalize_topic": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_outline": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_chapters": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "build_concept_graph": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_section_outline": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_remediation_outline": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_chapter_section": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "chapter_content_research": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_section_questions": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_topup_questions": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_weak_concept_questions": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "grade_assignment_answers": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "chat_reply": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "extension_plan": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "custom_export_clarify": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "custom_export_generate": {"provider": "opencode-go", "model": "mimo-v2.5"},
}

EMBEDDING_NODE = {"provider": "openrouter", "model": "nvidia/nemotron-3-embed-1b:free"}
