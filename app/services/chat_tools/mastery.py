from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Concept, Module
from app.services.chat_tools._authz import require_started_course
from app.services.chat_tools._errors import ChatToolError
from app.services.mastery import get_concept_statuses


def get_weak_concepts_by_course(db: Session, user_id: int, course_slug: str) -> dict[str, str]:
    course = require_started_course(db, user_id, course_slug)
    return get_concept_statuses(db, user_id, course.id)


def get_weak_concepts_by_module(db: Session, user_id: int, course_slug: str, module_id: int) -> dict[str, str]:
    course = require_started_course(db, user_id, course_slug)
    module = db.query(Module).filter_by(id=module_id, course_id=course.id).first()
    if module is None:
        raise ChatToolError(f"No module {module_id} in course '{course_slug}'.")
    chapter_ids = [c.id for c in db.query(Chapter).filter_by(module_id=module.id).all()]
    names_in_module = {
        name for cid in chapter_ids
        for name in db.scalars(select(Concept.name).where(Concept.chapter_id == cid)).all()
    }
    statuses = get_concept_statuses(db, user_id, course.id)
    return {name: status for name, status in statuses.items() if name in names_in_module}


def get_weak_concepts_by_chapter(db: Session, user_id: int, course_slug: str, chapter_id: int) -> dict[str, str]:
    course = require_started_course(db, user_id, course_slug)
    chapter = db.get(Chapter, chapter_id)
    if chapter is None:
        raise ChatToolError(f"No chapter {chapter_id}.")
    names_in_chapter = set(db.scalars(select(Concept.name).where(Concept.chapter_id == chapter_id)).all())
    statuses = get_concept_statuses(db, user_id, course.id)
    return {name: status for name, status in statuses.items() if name in names_in_chapter}


def get_recurring_weak_concepts(
    db: Session, user_id: int, course_slug: str, min_occurrences: int = 3,
) -> list[dict]:
    """Concepts that appear in ChapterContent.remediation_target_tags across
    >= min_occurrences versions of the same chapter (global v1 has no tags;
    v2+ are remediation rounds), aggregated across the whole course, ranked
    by total occurrences descending."""
    course = require_started_course(db, user_id, course_slug)
    chapter_ids = [
        c.id for c in db.scalars(
            select(Chapter).join(Module, Chapter.module_id == Module.id).where(Module.course_id == course.id)
        ).all()
    ]
    tag_counts: dict[str, int] = {}
    for chapter_id in chapter_ids:
        contents = db.scalars(
            select(ChapterContent).where(
                ChapterContent.chapter_id == chapter_id,
                (ChapterContent.scope == "global")
                | ((ChapterContent.scope == "user") & (ChapterContent.user_id == user_id)),
            ).order_by(ChapterContent.version)
        ).all()
        for content in contents:
            for tag in (content.remediation_target_tags or []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return sorted(
        (
            {"concept_tag": tag, "occurrences": count}
            for tag, count in tag_counts.items() if count >= min_occurrences
        ),
        key=lambda row: -row["occurrences"],
    )
