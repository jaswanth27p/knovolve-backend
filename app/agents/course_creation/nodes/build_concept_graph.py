from pydantic import BaseModel
from app.llm.factory import get_chat_model
from app.llm.prompts import BUILD_CONCEPT_GRAPH_PROMPT
from app.llm.retry import call_with_retry
from app.agents.course_creation.state import CourseCreationState


class ConceptDraft(BaseModel):
    name: str
    chapter_title: str


class ConceptEdgeDraft(BaseModel):
    concept_name: str
    prerequisite_name: str


class ConceptGraphResponse(BaseModel):
    concepts: list[ConceptDraft]
    edges: list[ConceptEdgeDraft]


def build_concept_graph(state: CourseCreationState) -> CourseCreationState:
    model = get_chat_model("build_concept_graph")
    structured = model.with_structured_output(ConceptGraphResponse)

    # generate_chapters always runs before build_concept_graph (see graph.py's
    # edge wiring), so modules is guaranteed populated by this point.
    modules = state["modules"]
    assert modules is not None
    chapters_summary = "\n".join(
        f"- {ch['title']} (module: {m['title']})"
        for m in modules for ch in m["chapters"]
    )
    messages = BUILD_CONCEPT_GRAPH_PROMPT.format_messages(chapters_summary=chapters_summary)
    result = call_with_retry(structured.invoke, messages)
    concepts = result.concepts if isinstance(result, ConceptGraphResponse) else result
    edges = result.edges if isinstance(result, ConceptGraphResponse) else result
    return {
        **state,
        "concepts": [c.model_dump() for c in concepts],
        "concept_edges": [e.model_dump() for e in edges],
    }
