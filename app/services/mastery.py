"""Weak/strong concept detection, derived entirely from existing state — no
new table, no separate write path. A chapter contributes its weak concepts
only while it is NOT YET passed (app.services.progression); the moment
chapter_passed turns true, it stops contributing anything, permanently, by
design. See design doc 2026-09-08-02-weak-strong-detection-design.md."""
from collections.abc import Sequence

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


def choose_relevant_contents(contents: Sequence[ChapterContent]) -> dict[int, ChapterContent]:
    """Pure equivalent of ``progression._resolve_relevant_content`` over a batch
    already fetched by the caller: each chapter's own latest ``scope="user"``
    version wins over its ``scope="global"`` version."""
    chosen: dict[int, ChapterContent] = {}
    for content in contents:
        current = chosen.get(content.chapter_id)
        if current is None or (
            content.scope == "user" and (current.scope != "user" or content.version > current.version)
        ):
            chosen[content.chapter_id] = content
    return chosen


def get_concept_statuses_for_courses(
    db: Session, user_id: int, course_ids: list[int]
) -> dict[int, dict[str, str]]:
    """Batched ``get_concept_statuses`` over many courses in a constant number
    of queries. See :func:`get_concept_status_counts_for_courses`."""
    if not course_ids:
        return {}
    rows = db.execute(
        select(Chapter.id, Module.course_id)
        .join(Module, Chapter.module_id == Module.id)
        .where(
            Module.course_id.in_(course_ids),
            or_(Module.scope == "global", and_(Module.scope == "user", Module.user_id == user_id)),
        )
    ).all()
    chapters_by_course: dict[int, list[int]] = {}
    for chapter_id, course_id in rows:
        chapters_by_course.setdefault(course_id, []).append(chapter_id)
    all_chapter_ids = [cid for ids in chapters_by_course.values() for cid in ids]
    if not all_chapter_ids:
        return {course_id: {} for course_id in course_ids}

    contents = db.scalars(
        select(ChapterContent).where(
            ChapterContent.chapter_id.in_(all_chapter_ids),
            or_(
                ChapterContent.scope == "global",
                and_(ChapterContent.scope == "user", ChapterContent.user_id == user_id),
            ),
        )
    ).all()
    chosen = choose_relevant_contents(contents)
    passed = progression.passed_chapter_ids(db, user_id, all_chapter_ids)

    # Weak tags are per-course: concept names can collide across courses, and a
    # weak tag in one course must not colour a same-named concept in another.
    weak_tags_by_course: dict[int, set[str]] = {}
    for course_id, ids in chapters_by_course.items():
        tags: set[str] = set()
        for chapter_id in ids:
            if chapter_id in passed:
                continue
            content = chosen.get(chapter_id)
            if content is not None and content.remediation_target_tags:
                tags.update(content.remediation_target_tags)
        weak_tags_by_course[course_id] = tags

    concept_rows = db.execute(
        select(Concept.name, Concept.chapter_id).where(Concept.chapter_id.in_(all_chapter_ids))
    ).all()
    names_by_chapter: dict[int, list[str]] = {}
    for name, chapter_id in concept_rows:
        names_by_chapter.setdefault(chapter_id, []).append(name)

    result: dict[int, dict[str, str]] = {}
    for course_id in course_ids:
        weak_tags = weak_tags_by_course.get(course_id, set())
        statuses: dict[str, str] = {}
        for chapter_id in chapters_by_course.get(course_id, []):
            concept_names = names_by_chapter.get(chapter_id)
            if not concept_names:
                continue
            if chosen.get(chapter_id) is None:
                for name in concept_names:
                    statuses[name] = "unassessed"
                continue
            for name in concept_names:
                statuses[name] = "weak" if name in weak_tags else "strong"
        result[course_id] = statuses
    return result


def get_concept_status_counts_for_courses(
    db: Session, user_id: int, course_ids: list[int]
) -> dict[int, tuple[int, int]]:
    """Per-course ``(weak_count, strong_count)`` equivalent to counting the
    values of :func:`get_concept_statuses`, computed for all courses in a
    constant number of queries (no per-chapter round trips)."""
    statuses_by_course = get_concept_statuses_for_courses(db, user_id, course_ids)
    return {
        course_id: (
            sum(1 for status in statuses.values() if status == "weak"),
            sum(1 for status in statuses.values() if status == "strong"),
        )
        for course_id, statuses in statuses_by_course.items()
    }


def get_concept_statuses(db: Session, user_id: int, course_id: int) -> dict[str, str]:
    return get_concept_statuses_for_courses(db, user_id, [course_id]).get(course_id, {})


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
