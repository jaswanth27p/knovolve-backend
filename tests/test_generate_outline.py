from unittest.mock import patch, MagicMock
from app.agents.course_creation.state import CourseCreationState
from app.agents.course_creation.nodes.generate_outline import generate_outline, ModuleDraft


def test_generate_outline_produces_validated_modules():
    state: CourseCreationState = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": "typescript",
              "topic_embedding": [0.0], "existing_course_id": None,
              "modules": None, "concepts": None, "concept_edges": None, "error": None}

    fake_modules = [
        ModuleDraft(title="Basic Types", objective="Understand primitives", order=1),
        ModuleDraft(title="Generics", objective="Write reusable typed code", order=2),
    ]
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value.invoke.return_value = fake_modules

    with patch("app.agents.course_creation.nodes.generate_outline.get_chat_model", return_value=mock_model):
        result = generate_outline(state)

    modules = result["modules"]
    assert modules is not None  # generate_outline always populates modules on success
    assert len(modules) == 2
    assert modules[0]["title"] == "Basic Types"
    assert modules[0]["chapters"] == []
