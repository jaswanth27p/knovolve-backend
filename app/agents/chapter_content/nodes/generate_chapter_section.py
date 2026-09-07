from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_CHAPTER_SECTION_PROMPT
from app.llm.retry import call_with_retry


class ExampleDraft(BaseModel):
    prompt: str
    walkthrough: str


class DiagramNodeDraft(BaseModel):
    id: str
    label: str


class DiagramEdgeDraft(BaseModel):
    source: str
    target: str
    label: str | None = None


class DiagramSpecDraft(BaseModel):
    nodes: list[DiagramNodeDraft]
    edges: list[DiagramEdgeDraft]


class ChapterSectionResponse(BaseModel):
    body_markdown: str
    examples: list[ExampleDraft] = Field(default_factory=list)
    diagram_spec: DiagramSpecDraft | None = None


_CORRECTIVE_MESSAGE = HumanMessage(
    content=(
        "Your previous response had zero examples, but this section's kind "
        "is \"teaching\" — it MUST include at least one example with a "
        "prompt and a full walkthrough. Regenerate the full response, "
        "including at least one example this time."
    )
)


def generate_chapter_section(
    chapter_title: str, chapter_objective: str, heading: str, objective: str, kind: str,
) -> ChapterSectionResponse:
    model = get_chat_model("generate_chapter_section")
    structured = model.with_structured_output(ChapterSectionResponse)
    messages = GENERATE_CHAPTER_SECTION_PROMPT.format_messages(
        chapter_title=chapter_title, chapter_objective=chapter_objective,
        heading=heading, objective=objective, kind=kind,
    )
    result = call_with_retry(structured.invoke, messages)

    if kind == "teaching" and not result.examples:
        result = call_with_retry(structured.invoke, [*messages, _CORRECTIVE_MESSAGE])
        if not result.examples:
            raise ValueError(
                f"generate_chapter_section: LLM produced zero examples for "
                f"teaching section {heading!r} after a corrective retry"
            )
    return result
