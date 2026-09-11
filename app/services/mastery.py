"""Weak/strong concept detection, derived entirely from existing state — no
new table, no separate write path. A chapter contributes its weak concepts
only while it is NOT YET passed (app.services.progression); the moment
chapter_passed turns true, it stops contributing anything, permanently, by
design. See design doc 2026-09-08-02-weak-strong-detection-design.md."""
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Concept, Module
from app.services import progression

def _visible_chapter_ids(db: Session, user_id: int, course_id: int) -> list[int]:
    return [
        c.id for c in db.scalars(
            select(Chapter).join(Module, Chapter.module_id == Module.id).where(
                Module.course_id == course_id,
                or_(Module.scope == "global", and_(Module.scope == "user", Module.user_id == user_id)),
            )
        ).all()
    ]


def get_weak_concept_tags(db: Session, user_id: int, course_id: int) -> set[str]:
    chapter_ids = _visible_chapter_ids(db, user_id, course_id)
    tags: set[str] = set()
    for chapter_id in chapter_ids:
        if progression._chapter_passed(db, chapter_id, user_id):
            continue
        content = progression._resolve_relevant_content(db, chapter_id, user_id)
        if content is None or not content.remediation_target_tags:
            continue
        tags.update(content.remediation_target_tags)
    return tags


def get_concept_statuses(db: Session, user_id: int, course_id: int) -> dict[str, str]:
    chapter_ids = _visible_chapter_ids(db, user_id, course_id)
    weak_tags = get_weak_concept_tags(db, user_id, course_id)

    statuses: dict[str, str] = {}
    for chapter_id in chapter_ids:
        concept_names = db.scalars(select(Concept.name).where(Concept.chapter_id == chapter_id)).all()
        if not concept_names:
            continue
        if progression._resolve_relevant_content(db, chapter_id, user_id) is None:
            status = "unassessed"
            for name in concept_names:
                statuses[name] = status
            continue
        for name in concept_names:
            statuses[name] = "weak" if name in weak_tags else "strong"
    return statuses


def get_module_lagged_history(db: Session, user_id: int, module_id: int) -> list[dict]:
    chapter_ids = [c.id for c in db.scalars(select(Chapter).where(Chapter.module_id == module_id)).all()]
    if not chapter_ids:
        return []

    contents = db.scalars(
        select(ChapterContent).where(
            ChapterContent.chapter_id.in_(chapter_ids),
            (ChapterContent.scope == "global")
            | ((ChapterContent.scope == "user") & (ChapterContent.user_id == user_id)),
        )
    ).all()

    history: list[dict] = []
    for content in contents:
        if not content.remediation_target_tags:
            continue
        history.append({
            "chapter_id": content.chapter_id,
            "version": content.version,
            "concept_tags": content.remediation_target_tags,
            "triggered_by_attempt_id": content.remediation_source_attempt_id,
            "created_at": content.created_at,
        })
    return history
