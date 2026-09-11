"""Idempotent content-completion primitives for the full-course run.

This module deliberately mirrors the streaming chapter generator's database
rules without yielding HTTP events. It is safe to call repeatedly and safe to
call while a chapter page is generating the same content: already-persisted
sections are adopted, not duplicated.
"""
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.agents.assignment.generate import generate_chapter_assignment, generate_module_assignment
from app.agents.chapter_content.generate import _get_or_create_content
from app.agents.chapter_content.nodes.generate_chapter_section import generate_chapter_section
from app.agents.chapter_content.nodes.generate_section_outline import generate_section_outline
from app.agents.chapter_content.remediate import remediate_chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter
from app.tasks.render_diagram_task import render_and_upload_diagram


def _now():
    return datetime.now(timezone.utc)


def _diagram_spec_dict(spec):
    """Normalize `generate_chapter_section`'s diagram spec to a JSONB-ready
    dict. Real runs return a `DiagramSpecDraft` pydantic model; tests stub the
    boundary with a plain dict, so accept both."""
    if spec is None or isinstance(spec, dict):
        return spec
    return spec.model_dump()


def _finalize_diagram(db: Session, section: ChapterContentSection) -> None:
    if section.diagram_spec is None or section.diagram_status != "pending":
        return
    try:
        url = render_and_upload_diagram(section)
    except Exception:
        section.diagram_status = "failed"
        db.commit()
        return
    section.diagram_status = "ready"
    section.diagram_image_url = url
    db.commit()


def _finalize_pending_diagrams(db: Session, content: ChapterContent) -> None:
    """Complete diagrams the streaming generator left pending (or that a prior
    run failed to finish) even when the chapter content is already `ready` — a
    ready row with `diagram_status="pending"` is otherwise never revisited."""
    sections = db.scalars(
        select(ChapterContentSection).where(
            ChapterContentSection.chapter_content_id == content.id,
            ChapterContentSection.diagram_status == "pending",
        )
    ).all()
    for section in sections:
        _finalize_diagram(db, section)


def ensure_chapter_content(db: Session, chapter: Chapter) -> None:
    content = _get_or_create_content(db, chapter)
    if content.status == "ready":
        _finalize_pending_diagrams(db, content)
        return
    if content.remediation_source_attempt_id is not None:
        return
    if not content.outline:
        outline = generate_section_outline(chapter.title, chapter.objective)
        db.refresh(content)
        if not content.outline:
            content.outline = [entry.model_dump() for entry in outline]
            content.updated_at = _now()
            db.commit()
    done_orders = {
        order
        for order, in db.query(ChapterContentSection.order)
        .filter_by(chapter_content_id=content.id)
        .all()
    }
    for order, entry in enumerate(content.outline):
        if order in done_orders:
            continue
        result = generate_chapter_section(
            chapter.title, chapter.objective, entry["heading"], entry["objective"], entry["kind"]
        )
        diagram_spec = _diagram_spec_dict(result.diagram_spec)
        section = ChapterContentSection(
            chapter_content_id=content.id,
            order=order,
            heading=entry["heading"],
            kind=entry["kind"],
            body_markdown=result.body_markdown,
            examples=[example.model_dump() for example in result.examples],
            diagram_spec=diagram_spec,
            diagram_status="pending" if diagram_spec is not None else None,
        )
        db.add(section)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            section = db.scalar(
                select(ChapterContentSection).where(
                    ChapterContentSection.chapter_content_id == content.id,
                    ChapterContentSection.order == order,
                )
            )
            if section is None:
                raise
        else:
            db.refresh(section)
            _finalize_diagram(db, section)
    persisted = {
        order
        for order, in db.query(ChapterContentSection.order)
        .filter_by(chapter_content_id=content.id)
        .all()
    }
    if persisted == set(range(len(content.outline))):
        content.status = "ready"
        content.updated_at = _now()
        db.commit()
        _finalize_pending_diagrams(db, content)


def ensure_chapter_assignment(db: Session, content: ChapterContent) -> None:
    if content.status != "ready":
        raise ValueError("Chapter assignment requires ready content")
    generate_chapter_assignment(content.id, db)


def ensure_remediation_content(db: Session, chapter_id: int, user_id: int, content: ChapterContent) -> None:
    if content.chapter_id != chapter_id:
        raise ValueError("Remediation content does not belong to this chapter")
    if content.scope != "user" or content.user_id != user_id:
        raise ValueError("Remediation content is not owned by this user")
    if content.remediation_source_attempt_id is None:
        raise ValueError("Only triggered remediation versions can be completed")
    if content.status == "ready":
        _finalize_pending_diagrams(db, content)
        return
    remediate_chapter(
        chapter_id,
        user_id,
        content.remediation_target_tags or [],
        content.remediation_source_attempt_id,
        db,
    )


def ensure_module_assignment(db: Session, module_id: int) -> None:
    generate_module_assignment(module_id, db)
