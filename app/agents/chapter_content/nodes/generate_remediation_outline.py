from app.agents.chapter_content.nodes.generate_section_outline import SectionOutlineDraft, SectionOutlineResponse
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_REMEDIATION_OUTLINE_PROMPT
from app.llm.retry import call_with_retry


def generate_remediation_outline(
    chapter_title: str, chapter_objective: str, weak_concept_tags: list[str],
) -> list[SectionOutlineDraft]:
    model = get_chat_model("generate_remediation_outline")
    structured = model.with_structured_output(SectionOutlineResponse)
    messages = GENERATE_REMEDIATION_OUTLINE_PROMPT.format_messages(
        chapter_title=chapter_title, chapter_objective=chapter_objective,
        weak_concept_tags=", ".join(weak_concept_tags),
    )
    result = call_with_retry(structured.invoke, messages)
    sections = result.sections if isinstance(result, SectionOutlineResponse) else result
    if not sections:
        raise ValueError("generate_remediation_outline: LLM produced zero sections")
    return sorted(sections, key=lambda s: s.order)[:3]
