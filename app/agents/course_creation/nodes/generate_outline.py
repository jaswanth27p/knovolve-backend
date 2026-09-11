from pydantic import BaseModel
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_OUTLINE_PROMPT, WEB_RESEARCH_PROMPT
from app.llm.retry import call_with_retry
from app.llm.web_research import FALLBACK_RESEARCH_NOTES, run_web_research
from app.agents.course_creation.state import CourseCreationState


class ModuleDraft(BaseModel):
    title: str
    objective: str
    order: int


class OutlineResponse(BaseModel):
    modules: list[ModuleDraft]


def generate_outline(state: CourseCreationState) -> CourseCreationState:
    model = get_chat_model("generate_outline")
    research_notes = run_web_research(
        model, WEB_RESEARCH_PROMPT.format_messages(topic_raw=state["topic_raw"])
    )
    structured = model.with_structured_output(OutlineResponse)
    messages = GENERATE_OUTLINE_PROMPT.format_messages(
        topic_raw=state["topic_raw"],
        research_notes=research_notes or FALLBACK_RESEARCH_NOTES,
    )
    result = call_with_retry(structured.invoke, messages)
    modules = [
        {**m.model_dump(), "order": idx, "chapters": []}
        for idx, m in enumerate(
            result.modules if isinstance(result, OutlineResponse) else result, start=1
        )
    ]
    return {**state, "modules": modules}
