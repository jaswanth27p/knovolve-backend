LLM_NODES = {
    "canonicalize_topic": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_outline": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "generate_chapters": {"provider": "opencode-go", "model": "mimo-v2.5"},
    "build_concept_graph": {"provider": "opencode-go", "model": "mimo-v2.5"},
}

EMBEDDING_NODE = {"provider": "openrouter", "model": "nvidia/nemotron-3-embed-1b:free"}
