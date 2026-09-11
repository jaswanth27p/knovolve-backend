"""One chapter-scoped web-research pass per content version.

Runs after a version's section outline is persisted and before its section
bodies are written, so the brief can target the planned headings (and, for
remediation, the weak concepts being re-taught). Best-effort: any failure
degrades to no notes and generation continues.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.llm.factory import get_chat_model
from app.llm.prompts import CHAPTER_RESEARCH_PROMPT
from app.llm.web_research import run_web_research
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter

logger = logging.getLogger(__name__)


def ensure_chapter_research(db: Session, chapter: Chapter, content: ChapterContent) -> None:
    if not settings.chapter_research_enabled:
        return
    if content.research_notes:
        return
    if not content.outline:
        return

    headings = [str(entry.get("heading", "")) for entry in content.outline]
    try:
        model = get_chat_model("chapter_content_research")
        notes = run_web_research(
            model,
            CHAPTER_RESEARCH_PROMPT.format_messages(
                chapter_title=chapter.title,
                chapter_objective=chapter.objective,
                section_headings="\n".join(f"- {h}" for h in headings),
                weak_concepts=", ".join(content.remediation_target_tags or []) or "(none)",
            ),
            enabled=True,
            max_rounds=settings.chapter_research_max_tool_rounds,
            max_tool_calls=settings.chapter_research_max_tool_calls,
        )
    except Exception:
        logger.warning("chapter research failed; continuing without it", exc_info=True)
        return

    if not notes:
        return
    db.refresh(content)
    if content.research_notes:
        return  # a concurrent generator already stored one
    content.research_notes = notes
    content.updated_at = datetime.now(timezone.utc)
    db.commit()
