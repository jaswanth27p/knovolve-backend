from pydantic import BaseModel
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_CHAPTERS_PROMPT
from app.llm.retry import call_with_retry
from app.agents.course_creation.state import CourseCreationState


class ChapterDraft(BaseModel):
    title: str
    objective: str
    order: int


class ChaptersResponse(BaseModel):
    chapters: list[ChapterDraft]


def _existing_titles(modules: list) -> str:
    titles = [ch["title"] for m in modules for ch in m.get("chapters", [])]
    return "\n".join(f"- {t}" for t in titles) if titles else "(none)"


def generate_chapters(state: CourseCreationState) -> CourseCreationState:
    """Per-module loop. Each module's chapters are generated and stored as
    soon as that module completes — combined with the LangGraph Postgres
    checkpointer (Task 8's graph wiring), this means a crash after N of M
    modules resumes at module N+1 rather than regenerating everything.

    Two collision-safety behaviours:
    - every module is prompted with the chapter titles already produced for
      *other* modules, so titles stay unique across the whole course;
    - when `validate_course` flags a cross-module duplicate (state carries a
      `rerun_modules` hint), only those modules are cleared and regenerated —
      untouched modules keep their already-generated chapters, so the repair
      never regenerates the whole course.
    """
    model = get_chat_model("generate_chapters")
    structured = model.with_structured_output(ChaptersResponse)

    modules = state["modules"]
    assert modules is not None
    rerun = set(state.get("rerun_modules") or [])

    # Process in module order so earlier modules' titles feed the constraints
    # of later ones. Regenerate targets are matched by module title, which is
    # unique by construction (generate_outline scopes modules distinctly).
    for module in modules:
        if module["chapters"] and module["title"] not in rerun:
            continue  # already generated in a prior (possibly crashed) run
        module["chapters"] = []
        messages = GENERATE_CHAPTERS_PROMPT.format_messages(
            module_title=module["title"],
            module_objective=module["objective"],
            existing_titles=_existing_titles(
                [m for m in modules if m is not module]
            ),
        )
        result = call_with_retry(structured.invoke, messages)
        chapters = result.chapters if isinstance(result, ChaptersResponse) else result
        module["chapters"] = [
            {**c.model_dump(), "order": idx} for idx, c in enumerate(chapters, start=1)
        ]

    return {**state, "modules": modules, "rerun_modules": []}

