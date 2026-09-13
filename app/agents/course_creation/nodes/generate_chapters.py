import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
from app.config import settings
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_CHAPTERS_PROMPT
from app.llm.retry import call_with_retry
from app.llm.web_research import FALLBACK_RESEARCH_NOTES, fetch_structure_research
from app.agents.course_creation.state import CourseCreationState

logger = logging.getLogger(__name__)


class ChapterDraft(BaseModel):
    title: str
    objective: str
    order: int


class ChaptersResponse(BaseModel):
    chapters: list[ChapterDraft]


def _existing_titles(modules: list) -> str:
    titles = [ch["title"] for m in modules for ch in m.get("chapters", [])]
    return "\n".join(f"- {t}" for t in titles) if titles else "(none)"


def _generate_module_chapters(module: dict, existing_titles: str) -> list[dict]:
    """Generate one module's chapters. Runs in a worker thread (one per
    module), so it builds its own model/client rather than sharing one across
    threads."""
    model = get_chat_model("generate_chapters")
    structured = model.with_structured_output(ChaptersResponse)
    research_notes = fetch_structure_research(
        f"{module['title']} — {module['objective']}"
    )
    messages = GENERATE_CHAPTERS_PROMPT.format_messages(
        module_title=module["title"],
        module_objective=module["objective"],
        existing_titles=existing_titles,
        research_notes=research_notes or FALLBACK_RESEARCH_NOTES,
    )
    result = call_with_retry(structured.invoke, messages)
    chapters = result.chapters if isinstance(result, ChaptersResponse) else result
    return [
        {**c.model_dump(), "order": idx} for idx, c in enumerate(chapters, start=1)
    ]


def generate_chapters(state: CourseCreationState) -> CourseCreationState:
    """Fan out one worker per module (bounded by
    ``course_structure_max_workers``), each grounded in a one-shot web research
    pass. Resumability and targeted rerun behavior are unchanged: a module with
    chapters already populated (and not flagged for rerun) is skipped.

    Cross-module title uniqueness is still enforced by ``validate_course``; on
    a duplicate, ``rerun_modules`` regenerates just the colliding modules (now
    sequentially visible to each other because the surviving modules' titles are
    passed in as ``existing_titles``).
    """
    modules = state["modules"]
    assert modules is not None
    rerun = set(state.get("rerun_modules") or [])

    targets = [m for m in modules if not m["chapters"] or m["title"] in rerun]
    for module in targets:
        module["chapters"] = []

    if not targets:
        return {**state, "modules": modules, "rerun_modules": []}

    # Cold-start modules are generated in parallel; `validate_course`'s
    # collision reruns are generated SEQUENTIALLY, because the whole point of a
    # rerun is that sibling modules already collide — running them in parallel
    # again would reproduce the collision (each would still not see the other's
    # fresh titles).
    cold_targets = [m for m in targets if m["title"] not in rerun]
    rerun_targets = [m for m in targets if m["title"] in rerun]

    if cold_targets:
        cold_ids = {id(m) for m in cold_targets}
        existing_titles = _existing_titles(
            [m for m in modules if id(m) not in cold_ids]
        )
        max_workers = max(1, min(settings.course_structure_max_workers, len(cold_targets)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    lambda module: _generate_module_chapters(module, existing_titles),
                    cold_targets,
                )
            )
        for module, chapters in zip(cold_targets, results):
            module["chapters"] = chapters

    if rerun_targets:
        logger.warning(
            "regenerating %d module(s) sequentially to resolve duplicate chapter "
            "titles: %s",
            len(rerun_targets),
            [m["title"] for m in rerun_targets],
        )
        for module in rerun_targets:
            module["chapters"] = _generate_module_chapters(
                module,
                _existing_titles([m for m in modules if m is not module]),
            )

    return {**state, "modules": modules, "rerun_modules": []}
