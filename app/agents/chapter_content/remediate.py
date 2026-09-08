"""Generates a SHORT, narrow chapter-version re-teach for one user's weak
concepts, triggered by grade_assignment_attempt's chapter-level dispatch
(Trigger A only). Unlike stream_chapter_content (V1, a live HTTP stream the
user waits on), this runs as a background Celery task with no client polling
it — idempotency is keyed on `remediation_source_attempt_id`, not per-section
streaming resume: one graded attempt triggers at most one new ChapterContent
version, and a redelivered task adopts the row a prior run already created
instead of minting a duplicate version.

No diagram generation for remediation sections in V1 (documented cut, not an
oversight): any diagram_spec generate_chapter_section returns is deliberately
dropped, never persisted or dispatched to render_diagram_task.
"""
from datetime import datetime, timezone
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.agents.chapter_content.nodes.generate_remediation_outline import generate_remediation_outline
from app.agents.chapter_content.nodes.generate_chapter_section import generate_chapter_section
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter
from app.tasks.assignment_tasks import generate_chapter_assignment_task


def _get_or_create_content(
    chapter_id: int, user_id: int, source_attempt_id: int, weak_concept_tags: list[str], db: Session,
) -> ChapterContent:
    existing = db.scalar(
        select(ChapterContent).where(ChapterContent.remediation_source_attempt_id == source_attempt_id)
    )
    if existing is not None:
        return existing

    max_version = db.scalar(
        select(func.max(ChapterContent.version)).where(
            ChapterContent.chapter_id == chapter_id,
            or_(
                ChapterContent.scope == "global",
                and_(ChapterContent.scope == "user", ChapterContent.user_id == user_id),
            ),
        )
    ) or 0

    now = datetime.now(timezone.utc)
    content = ChapterContent(
        chapter_id=chapter_id, version=max_version + 1, scope="user", user_id=user_id, status="generating",
        outline=[], remediation_target_tags=weak_concept_tags, remediation_source_attempt_id=source_attempt_id,
        created_at=now, updated_at=now,
    )
    db.add(content)
    try:
        db.commit()
    except IntegrityError:
        # A redelivered task (worker crash after this INSERT but before ack)
        # races its own retry here; the unique index on
        # remediation_source_attempt_id lets exactly one insert win.
        db.rollback()
        winner = db.scalar(
            select(ChapterContent).where(ChapterContent.remediation_source_attempt_id == source_attempt_id)
        )
        if winner is None:
            raise
        return winner
    db.refresh(content)
    return content


def remediate_chapter(
    chapter_id: int, user_id: int, weak_concept_tags: list[str], source_attempt_id: int, db: Session,
) -> None:
    content = _get_or_create_content(chapter_id, user_id, source_attempt_id, weak_concept_tags, db)
    if content.status in ("ready", "failed"):
        return  # already done (redelivery) or already failed — no retry-resume for remediation in V1

    chapter = db.get(Chapter, chapter_id)
    if chapter is None:
        raise ValueError(f"Chapter {chapter_id} not found")

    try:
        if not content.outline:
            outline = generate_remediation_outline(chapter.title, chapter.objective, weak_concept_tags)
            db.refresh(content)
            if not content.outline:
                content.outline = [o.model_dump() for o in outline]
                content.updated_at = datetime.now(timezone.utc)
                db.commit()

        existing_orders = {
            s.order for s in db.scalars(
                select(ChapterContentSection).where(ChapterContentSection.chapter_content_id == content.id)
            )
        }
        for i, entry in enumerate(content.outline):
            if i in existing_orders:
                continue
            result = generate_chapter_section(
                chapter.title, chapter.objective, entry["heading"], entry["objective"], entry["kind"],
            )
            section = ChapterContentSection(
                # kind is hardcoded, not passed through from entry["kind"]:
                # remediation sections are always re-teaching content by
                # definition, and generate_chapter_assignment only builds
                # questions from kind=="teaching" sections — pinning this
                # removes an LLM-controlled value from a place where the
                # answer is actually invariant.
                chapter_content_id=content.id, order=i, heading=entry["heading"], kind="teaching",
                body_markdown=result.body_markdown,
                examples=[e.model_dump() for e in result.examples],
                diagram_spec=None, diagram_status=None,  # no diagram generation for remediation, V1
            )
            db.add(section)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()  # a concurrent redelivery generated this exact section first

        content.status = "ready"
        content.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        db.rollback()
        content.status = "failed"
        content.error = str(exc)
        content.updated_at = datetime.now(timezone.utc)
        db.commit()
        return

    generate_chapter_assignment_task.delay(content.id)  # pyright: ignore[reportFunctionMemberAccess]
