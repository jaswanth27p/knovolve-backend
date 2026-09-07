from pydantic import BaseModel
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_OUTLINE_PROMPT
from app.agents.course_creation.state import CourseCreationState


class ModuleDraft(BaseModel):
    title: str
    objective: str
    order: int


class OutlineResponse(BaseModel):
    modules: list[ModuleDraft]


def generate_outline(state: CourseCreationState) -> CourseCreationState:
    model = get_chat_model("generate_outline")
    structured = model.with_structured_output(OutlineResponse)
    messages = GENERATE_OUTLINE_PROMPT.format_messages(topic_raw=state["topic_raw"])
    result = structured.invoke(messages)
    modules = [
        {**m.model_dump(), "chapters": []}
        for m in (result.modules if isinstance(result, OutlineResponse) else result)
    ]
    return {**state, "modules": modules}
