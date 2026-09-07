from app.agents.course_creation.nodes.validate_course import validate_course

def _base_state(**overrides):
    state = {"job_id": 1, "topic_raw": "T", "topic_slug": "t", "topic_embedding": [0.0],
              "existing_course_id": None,
              "modules": [{"title": "M1", "objective": "o", "order": 1,
                           "chapters": [{"title": "C1", "objective": "o", "order": 1}]}],
              "concepts": [{"name": "X", "chapter_title": "C1"}],
              "concept_edges": [], "error": None}
    state.update(overrides)
    return state

def test_valid_course_passes():
    result = validate_course(_base_state())
    assert result["error"] is None

def test_module_without_chapters_fails():
    state = _base_state(modules=[{"title": "M1", "objective": "o", "order": 1, "chapters": []}])
    result = validate_course(state)
    assert result["error"] is not None
    assert "chapter" in result["error"].lower()

def test_cyclic_prerequisite_fails():
    state = _base_state(
        concepts=[{"name": "A", "chapter_title": "C1"}, {"name": "B", "chapter_title": "C1"}],
        concept_edges=[{"concept_name": "A", "prerequisite_name": "B"},
                        {"concept_name": "B", "prerequisite_name": "A"}],
    )
    result = validate_course(state)
    assert result["error"] is not None
    assert "cycl" in result["error"].lower()

def test_dangling_concept_edge_reference_fails():
    state = _base_state(
        concepts=[{"name": "A", "chapter_title": "C1"}],
        concept_edges=[{"concept_name": "A", "prerequisite_name": "Nonexistent"}],
    )
    result = validate_course(state)
    assert result["error"] is not None
    assert "reference" in result["error"].lower() or "unknown" in result["error"].lower()
