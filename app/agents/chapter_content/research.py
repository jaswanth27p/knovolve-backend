"""One chapter-scoped web-research pass per content version.

Runs after a version's section outline is persisted and before its section
bodies are written. This is a single deterministic pass for latency: one
search, fetch the top URLs once, hand the extracted text to the section
writers. No agent tool loop, no retries, no summarizer model call. Best-effort:
any failure degrades to no notes and generation continues.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.llm.web_research import fetch_chapter_research
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

    query = f"{chapter.title} — {chapter.objective}".strip()
    tags = content.remediation_target_tags or []
    if tags:
        query = f"{query} ({', '.join(tags)})"
    query = query[:400]

    try:
        notes = fetch_chapter_research(
            query,
            top_urls=settings.chapter_research_top_urls,
            max_chars_per_page=settings.chapter_research_max_chars_per_page,
            max_total_chars=settings.chapter_research_max_total_chars,
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
