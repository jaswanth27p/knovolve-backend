from pydantic import BaseModel
from app.llm.factory import get_chat_model
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

    chapters_summary = "\n".join(
        f"- {ch['title']} (module: {m['title']})"
        for m in state["modules"] for ch in m["chapters"]
    )
    result = structured.invoke(
        f"Given these chapters, list the key concepts taught (each tied to one "
        f"chapter title) and prerequisite relationships between concepts:\n{chapters_summary}"
    )
    return {
        **state,
        "concepts": [c.model_dump() for c in result.concepts],
        "concept_edges": [e.model_dump() for e in result.edges],
    }
