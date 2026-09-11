from datetime import datetime, timezone
from unittest.mock import patch
from app.config import settings
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.agents.chapter_content import research as research_module


def _now():
    return datetime.now(timezone.utc)


def _make_chapter_with_content(db, slug, outline, tags=None, notes=None):
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
    db.add(course)
    db.flush()
    module = Module(course_id=course.id, title="M", objective="o", order=1)
    db.add(module)
    db.flush()
    chapter = Chapter(module_id=module.id, title="Recursion", objective="Understand recursion", order=1)
    db.add(chapter)
    db.flush()
    content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="generating",
                             outline=outline, remediation_target_tags=tags,
                             research_notes=notes, created_at=_now(), updated_at=_now())
    db.add(content)
    db.commit()
    db.refresh(chapter)
    db.refresh(content)
    return chapter, content


_OUTLINE = [{"heading": "Base cases", "objective": "o", "kind": "teaching"}]


def test_persists_notes_and_seeds_prompt_from_outline_and_tags():
    with SessionLocal() as db:
        chapter, content = _make_chapter_with_content(
            db, "cc-research-persist", _OUTLINE, tags=["recursion-base-case"]
        )
        with patch("app.agents.chapter_content.research.get_chat_model"), \
             patch("app.agents.chapter_content.research.run_web_research", return_value="FACT | http://src") as mock_research:
            research_module.ensure_chapter_research(db, chapter, content)

        messages = mock_research.call_args.args[1]
        blob = "\n".join(m.content for m in messages)
        assert "- Base cases" in blob
        assert "recursion-base-case" in blob
        db.refresh(content)
        assert content.research_notes == "FACT | http://src"


def test_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "chapter_research_enabled", False)
    with SessionLocal() as db:
        chapter, content = _make_chapter_with_content(db, "cc-research-disabled", _OUTLINE)
        with patch("app.agents.chapter_content.research.run_web_research") as mock_research:
            research_module.ensure_chapter_research(db, chapter, content)
        mock_research.assert_not_called()
        db.refresh(content)
        assert content.research_notes is None


def test_noop_when_notes_already_present():
    with SessionLocal() as db:
        chapter, content = _make_chapter_with_content(
            db, "cc-research-present", _OUTLINE, notes="already"
        )
        with patch("app.agents.chapter_content.research.run_web_research") as mock_research:
            research_module.ensure_chapter_research(db, chapter, content)
        mock_research.assert_not_called()


def test_noop_when_outline_empty():
    with SessionLocal() as db:
        chapter, content = _make_chapter_with_content(db, "cc-research-no-outline", [])
        with patch("app.agents.chapter_content.research.run_web_research") as mock_research:
            research_module.ensure_chapter_research(db, chapter, content)
        mock_research.assert_not_called()


def test_empty_result_leaves_notes_none():
    with SessionLocal() as db:
        chapter, content = _make_chapter_with_content(db, "cc-research-empty", _OUTLINE)
        with patch("app.agents.chapter_content.research.get_chat_model"), \
             patch("app.agents.chapter_content.research.run_web_research", return_value=""):
            research_module.ensure_chapter_research(db, chapter, content)
        db.refresh(content)
        assert content.research_notes is None


def test_research_failure_is_swallowed():
    with SessionLocal() as db:
        chapter, content = _make_chapter_with_content(db, "cc-research-fail", _OUTLINE)
        with patch("app.agents.chapter_content.research.get_chat_model"), \
             patch("app.agents.chapter_content.research.run_web_research", side_effect=RuntimeError("down")):
            research_module.ensure_chapter_research(db, chapter, content)  # must not raise
        db.refresh(content)
        assert content.research_notes is None
