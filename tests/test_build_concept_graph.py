from unittest.mock import patch, MagicMock
from app.agents.course_creation.state import CourseCreationState
from app.agents.course_creation.nodes.build_concept_graph import build_concept_graph, ConceptGraphResponse, ConceptDraft, ConceptEdgeDraft

def test_build_concept_graph_produces_concepts_and_edges():
    state: CourseCreationState = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": "typescript",
              "topic_embedding": [0.0], "existing_course_id": None,
              "modules": [{"title": "Generics", "objective": "...", "order": 1,
                           "chapters": [{"title": "Generic Functions", "objective": "...", "order": 1}]}],
              "concepts": None, "concept_edges": None, "error": None}

    fake_response = ConceptGraphResponse(
        concepts=[ConceptDraft(name="Generics", chapter_title="Generic Functions"),
                  ConceptDraft(name="Functions", chapter_title="Generic Functions")],
        edges=[ConceptEdgeDraft(concept_name="Generics", prerequisite_name="Functions")],
    )
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_response

    with patch("app.agents.course_creation.nodes.build_concept_graph.get_chat_model", return_value=mock_model):
        result = build_concept_graph(state)

    concepts = result["concepts"]
    concept_edges = result["concept_edges"]
    assert concepts is not None  # build_concept_graph always populates concepts
    assert concept_edges is not None  # build_concept_graph always populates concept_edges
    assert len(concepts) == 2
    assert concept_edges[0]["prerequisite_name"] == "Functions"
