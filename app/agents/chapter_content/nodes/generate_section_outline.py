from typing import Literal
from pydantic import BaseModel
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_SECTION_OUTLINE_PROMPT
from app.llm.retry import call_with_retry


class SectionOutlineDraft(BaseModel):
    heading: str
    objective: str
    kind: Literal["intro", "teaching"]
    order: int


class SectionOutlineResponse(BaseModel):
    sections: list[SectionOutlineDraft]


def generate_section_outline(chapter_title: str, chapter_objective: str) -> list[SectionOutlineDraft]:
    model = get_chat_model("generate_section_outline")
    structured = model.with_structured_output(SectionOutlineResponse)
    messages = GENERATE_SECTION_OUTLINE_PROMPT.format_messages(
        chapter_title=chapter_title, chapter_objective=chapter_objective,
    )
    result = call_with_retry(structured.invoke, messages)
    sections = result.sections if isinstance(result, SectionOutlineResponse) else result
    return sorted(sections, key=lambda s: s.order)
