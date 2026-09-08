"""Generates a chapter's assignment: one structured-output LLM call per
teaching section, topped up from a whole-chapter prompt if the total falls
short of MIN_QUESTIONS. Plain function, not a LangGraph graph — nothing
here is long-running or has an async sub-step (unlike chapter content's
diagram rendering), so a crash just retries the whole thing; there is
nothing slow enough mid-way to need per-step persistence.
"""

from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter
from app.agents.assignment.nodes.generate_section_questions import generate_questions_for_section
from app.agents.assignment.nodes.generate_topup_questions import generate_topup_questions

MIN_QUESTIONS = 3


def _get_or_create_assignment(
    db: Session, *, level: str, scope: str,
    chapter_content_id: int | None = None, module_id: int | None = None, user_id: int | None = None,
) -> tuple[Assignment, bool]:
    """Get-or-create against the partial unique indexes (Task 1). Returns
    (assignment, created) — created=False means a row already existed
    (any status) and the caller decides what to do about it."""
    filters = [Assignment.level == level, Assignment.scope == scope]
    if chapter_content_id is not None:
        filters.append(Assignment.chapter_content_id == chapter_content_id)
    if module_id is not None:
        filters.append(Assignment.module_id == module_id)
    if scope == "user":
        filters.append(Assignment.user_id == user_id)

    existing = db.scalar(select(Assignment).where(*filters))
    if existing is not None:
        return existing, False

    now = datetime.now(timezone.utc)
    assignment = Assignment(
        level=level, scope=scope, chapter_content_id=chapter_content_id, module_id=module_id,
        user_id=user_id, status="generating", created_at=now, updated_at=now,
    )
    db.add(assignment)
    try:
        db.commit()
    except IntegrityError:
        # Two concurrent triggers (e.g. the auto-chain firing twice under a
        # racing double-open of chapter content) can both pass the select
        # above; the partial unique index lets exactly one insert win.
        db.rollback()
        winner = db.scalar(select(Assignment).where(*filters))
        if winner is None:
            raise
        return winner, False
    db.refresh(assignment)
    return assignment, True


def generate_chapter_assignment(chapter_content_id: int, db: Session) -> None:
    content = db.get(ChapterContent, chapter_content_id)
    if content is None:
        raise ValueError(f"ChapterContent {chapter_content_id} not found")
    chapter = db.get(Chapter, content.chapter_id)

    assignment, created = _get_or_create_assignment(
        db, level="chapter", scope=content.scope, chapter_content_id=chapter_content_id,
        user_id=content.user_id if content.scope == "user" else None,
    )
    if not created:
        if assignment.status != "failed":
            return  # already ready or another generation is in flight
        assignment.status = "generating"
        assignment.error = None
        assignment.updated_at = datetime.now(timezone.utc)
        db.commit()

    sections = (
        db.query(ChapterContentSection)
        .filter_by(chapter_content_id=chapter_content_id, kind="teaching")
        .order_by(ChapterContentSection.order)
        .all()
    )

    try:
        drafts_by_section: list[tuple[int | None, list]] = []
        for section in sections:
            drafts = generate_questions_for_section(
                chapter.title, chapter.objective, section.heading, section.body_markdown, section.examples,
            )
            drafts_by_section.append((section.id, drafts))

        total = sum(len(d) for _, d in drafts_by_section)
        if total < MIN_QUESTIONS:
            needed = MIN_QUESTIONS - total
            topup = generate_topup_questions(
                chapter.title, chapter.objective,
                [{"heading": s.heading, "body_markdown": s.body_markdown} for s in sections],
                needed,
            )
            drafts_by_section.append((None, topup))
    except Exception as exc:
        assignment.status = "failed"
        assignment.error = str(exc)
        assignment.updated_at = datetime.now(timezone.utc)
        db.commit()
        return

    order = 0
    for section_id, drafts in drafts_by_section:
        for q in drafts:
            db.add(AssignmentQuestion(
                assignment_id=assignment.id, order=order, type=q.type, text=q.text, options=q.options,
                correct_answer=q.correct_answer, explanation=q.explanation, concept_tag=q.concept_tag,
                difficulty=q.difficulty, source_section_id=section_id,
            ))
            order += 1
    assignment.status = "ready"
    assignment.updated_at = datetime.now(timezone.utc)
    db.commit()
