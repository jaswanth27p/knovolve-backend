from typing import cast
from app.agents.course_creation.state import CourseCreationState
from app.agents.course_creation.nodes.validate_course import validate_course

def _base_state(**overrides: object) -> CourseCreationState:
    state: CourseCreationState = {"job_id": 1, "topic_raw": "T", "topic_slug": "t", "topic_embedding": [0.0],
              "existing_course_id": None,
              "modules": [{"title": "M1", "objective": "o", "order": 1,
                           "chapters": [{"title": "C1", "objective": "o", "order": 1}]}],
              "concepts": [{"name": "X", "chapter_title": "C1"}],
              "concept_edges": [], "error": None}
    # overrides is a caller-supplied **kwargs dict of valid CourseCreationState
    # fields (verified by every call site in this file); a TypedDict can't
    # express "partial update from arbitrary kwargs" any more precisely than
    # this cast.
    return cast(CourseCreationState, {**state, **overrides})

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


def test_dup_chapter_title_across_modules_fails_with_rerun_hint():
    """A chapter title appearing in more than one module makes concept
    references ambiguous (persist resolves by title), so it must be caught here
    and routed back to generate_chapters with the offending modules named."""
    state = _base_state(
        modules=[
            {"title": "M1", "objective": "o", "order": 1,
             "chapters": [{"title": "Introduction", "objective": "o", "order": 1}]},
            {"title": "M2", "objective": "o", "order": 2,
             "chapters": [{"title": "Introduction", "objective": "o", "order": 1}]},
        ],
        concepts=[{"name": "X", "chapter_title": "Introduction"}],
    )
    result = validate_course(state)
    assert result["error"] is not None
    assert "duplicate chapter title" in result["error"]
    assert (result.get("rerun_modules") or []) == ["M1", "M2"]


def test_concept_referencing_unknown_chapter_fails():
    state = _base_state(
        concepts=[{"name": "X", "chapter_title": "Never Exists"}],
    )
    result = validate_course(state)
    assert result["error"] is not None
    assert "unknown chapter" in result["error"]


def test_cycle_cases():
    """Regression coverage for the DFS cycle detector: 3-node cycle, self-loop,
    and the two non-cycles it must NOT flag (disconnected components and a
    converging DAG)."""
    def state_for(name, edges):
        concepts = [{"name": n, "chapter_title": "C1"} for n in name]
        return _base_state(concepts=concepts, concept_edges=edges)

    three_node_cycle = state_for(
        ["A", "B", "C"],
        [{"concept_name": "A", "prerequisite_name": "B"},
         {"concept_name": "B", "prerequisite_name": "C"},
         {"concept_name": "C", "prerequisite_name": "A"}],
    )
    assert validate_course(three_node_cycle)["error"] == "concept prerequisite graph is cyclic"

    self_loop = state_for(
        ["A"],
        [{"concept_name": "A", "prerequisite_name": "A"}],
    )
    assert validate_course(self_loop)["error"] == "concept prerequisite graph is cyclic"

    disconnected = state_for(
        ["A", "B"],
        [{"concept_name": "A", "prerequisite_name": "B"}],
    )
    assert validate_course(disconnected)["error"] is None

    converging = state_for(
        ["A", "B", "C"],
        [{"concept_name": "A", "prerequisite_name": "C"},
         {"concept_name": "B", "prerequisite_name": "C"}],
    )
    assert validate_course(converging)["error"] is None
