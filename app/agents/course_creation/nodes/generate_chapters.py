from pydantic import BaseModel
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_CHAPTERS_PROMPT
from app.agents.course_creation.state import CourseCreationState


class ChapterDraft(BaseModel):
    title: str
    objective: str
    order: int


class ChaptersResponse(BaseModel):
    chapters: list[ChapterDraft]


def generate_chapters(state: CourseCreationState) -> CourseCreationState:
    """Per-module loop. Each module's chapters are generated and stored as
    soon as that module completes — combined with the LangGraph Postgres
    checkpointer (Task 8's graph wiring), this means a crash after N of M
    modules resumes at module N+1 rather than regenerating everything."""
    model = get_chat_model("generate_chapters")
    structured = model.with_structured_output(ChaptersResponse)

    modules = state["modules"]
    for module in modules:
        if module["chapters"]:
            continue  # already generated in a prior (possibly crashed) run
        messages = GENERATE_CHAPTERS_PROMPT.format_messages(
            module_title=module["title"], module_objective=module["objective"]
        )
        result = structured.invoke(messages)
        chapters = result.chapters if isinstance(result, ChaptersResponse) else result
        module["chapters"] = [c.model_dump() for c in chapters]

    return {**state, "modules": modules}

