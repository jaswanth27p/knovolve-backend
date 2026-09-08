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


from app.models.course import Module


def _spread_by_concept(questions: list[AssignmentQuestion], count: int) -> list[AssignmentQuestion]:
    """Pick `count` questions favoring distinct concept_tags first, so a
    reused subset doesn't accidentally duplicate one concept and skip
    another the original assignment covered."""
    seen_tags: set[str] = set()
    picked: list[AssignmentQuestion] = []
    leftover: list[AssignmentQuestion] = []
    for q in questions:
        if q.concept_tag not in seen_tags:
            seen_tags.add(q.concept_tag)
            picked.append(q)
        else:
            leftover.append(q)
        if len(picked) == count:
            return picked
    return (picked + leftover)[:count]


def generate_module_assignment(module_id: int, db: Session) -> None:
    module = db.get(Module, module_id)
    if module is None:
        raise ValueError(f"Module {module_id} not found")

    assignment, created = _get_or_create_assignment(db, level="module", scope="global", module_id=module_id)
    if not created:
        if assignment.status != "failed":
            return
        assignment.status = "generating"
        assignment.error = None
        assignment.updated_at = datetime.now(timezone.utc)
        db.commit()

    chapters = db.query(Chapter).filter_by(module_id=module_id).order_by(Chapter.order).all()

    try:
        all_drafts: list[tuple[int | None, object]] = []
        for chapter in chapters:
            content = db.scalar(
                select(ChapterContent).where(
                    ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global",
                )
            )
            if content is None or content.status != "ready":
                raise ValueError(f"chapter {chapter.id} content is not ready")

            chapter_assignment = db.scalar(
                select(Assignment).where(
                    Assignment.level == "chapter", Assignment.scope == "global",
                    Assignment.chapter_content_id == content.id, Assignment.status == "ready",
                )
            )
            sections = (
                db.query(ChapterContentSection)
                .filter_by(chapter_content_id=content.id, kind="teaching")
                .order_by(ChapterContentSection.order)
                .all()
            )

            if chapter_assignment is not None:
                existing_questions = (
                    db.query(AssignmentQuestion)
                    .filter_by(assignment_id=chapter_assignment.id)
                    .order_by(AssignmentQuestion.order)
                    .all()
                )
                reuse_count = max(1, -(-len(existing_questions) // 2))  # ceil(n/2), min 1
                for q in _spread_by_concept(existing_questions, reuse_count):
                    all_drafts.append((q.source_section_id, q))

                if sections:
                    fresh = generate_questions_for_section(
                        chapter.title, chapter.objective, sections[0].heading,
                        sections[0].body_markdown, sections[0].examples,
                    )
                    if fresh:
                        all_drafts.append((sections[0].id, fresh[0]))
            else:
                for section in sections:
                    drafts = generate_questions_for_section(
                        chapter.title, chapter.objective, section.heading, section.body_markdown, section.examples,
                    )
                    for q in drafts:
                        all_drafts.append((section.id, q))

        if len(all_drafts) < MIN_QUESTIONS:
            needed = MIN_QUESTIONS - len(all_drafts)
            topup = generate_topup_questions(
                module.title, module.objective,
                [{"heading": c.title, "body_markdown": c.objective} for c in chapters],
                needed,
            )
            for q in topup:
                all_drafts.append((None, q))
    except Exception as exc:
        assignment.status = "failed"
        assignment.error = str(exc)
        assignment.updated_at = datetime.now(timezone.utc)
        db.commit()
        return

    for order, (section_id, q) in enumerate(all_drafts):
        db.add(AssignmentQuestion(
            assignment_id=assignment.id, order=order, type=q.type, text=q.text, options=q.options,
            correct_answer=q.correct_answer, explanation=q.explanation, concept_tag=q.concept_tag,
            difficulty=q.difficulty, source_section_id=section_id,
        ))
    assignment.status = "ready"
    assignment.updated_at = datetime.now(timezone.utc)
    db.commit()
