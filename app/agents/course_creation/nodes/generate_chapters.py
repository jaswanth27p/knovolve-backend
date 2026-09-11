from pydantic import BaseModel
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_CHAPTERS_PROMPT, WEB_RESEARCH_PROMPT
from app.llm.retry import call_with_retry
from app.llm.web_research import FALLBACK_RESEARCH_NOTES, run_web_research
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
    """Per-module loop, each module optionally grounded in a web research
    brief. Resumability and targeted rerun behavior are unchanged: a module
    with chapters already populated (and not flagged for rerun) is skipped, so
    resumed runs do not re-research filled modules."""
    model = get_chat_model("generate_chapters")
    structured = model.with_structured_output(ChaptersResponse)

    modules = state["modules"]
    assert modules is not None
    rerun = set(state.get("rerun_modules") or [])

    for module in modules:
        if module["chapters"] and module["title"] not in rerun:
            continue  # already generated in a prior (possibly crashed) run
        module["chapters"] = []
        research_notes = run_web_research(
            model,
            WEB_RESEARCH_PROMPT.format_messages(
                topic_raw=f"{module['title']} — {module['objective']}"
            ),
        )
        messages = GENERATE_CHAPTERS_PROMPT.format_messages(
            module_title=module["title"],
            module_objective=module["objective"],
            existing_titles=_existing_titles(
                [m for m in modules if m is not module]
            ),
            research_notes=research_notes or FALLBACK_RESEARCH_NOTES,
        )
        result = call_with_retry(structured.invoke, messages)
        chapters = result.chapters if isinstance(result, ChaptersResponse) else result
        module["chapters"] = [
            {**c.model_dump(), "order": idx} for idx, c in enumerate(chapters, start=1)
        ]

    return {**state, "modules": modules, "rerun_modules": []}

