"""Streams a chapter's V1 content, generating whatever's missing.

Not a LangGraph graph with a Postgres checkpointer (unlike course
creation): this is a single HTTP request the user is actively waiting on,
not a long-running background job that can crash without an active
request to resume from. Per-row idempotency in Postgres is the whole
resumability mechanism — a retried request (or a second concurrent open of
the same chapter) replays what's already persisted and only generates
what's missing, driven by the SAME persisted outline every time.
"""

from datetime import datetime, timezone
from typing import Iterator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.agents.chapter_content.nodes.generate_section_outline import generate_section_outline
from app.agents.chapter_content.nodes.generate_chapter_section import generate_chapter_section
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter
from app.tasks.assignment_tasks import generate_chapter_assignment_task
from app.tasks.chapter_content_tasks import remediate_chapter_task
from app.tasks.render_diagram_task import render_diagram_task


def _section_event(section: ChapterContentSection) -> dict:
    return {
        "type": "section_ready",
        "order": section.order,
        "heading": section.heading,
        "kind": section.kind,
        "body_markdown": section.body_markdown,
        "examples": section.examples,
        "diagram_status": section.diagram_status,
        "diagram_image_url": section.diagram_image_url,
    }


def _get_or_create_content(chapter_id: int, db: Session) -> ChapterContent:
    content = db.scalar(
        select(ChapterContent).where(ChapterContent.chapter_id == chapter_id, ChapterContent.scope == "global")
    )
    if content is not None:
        return content
    now = datetime.now(timezone.utc)
    content = ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="generating",
                              outline=[], created_at=now, updated_at=now)
    db.add(content)
    try:
        db.commit()
    except IntegrityError:
        # Two concurrent opens of an ungenerated chapter can both pass the
        # select above; the partial unique index (chapter_id, version) for
        # scope="global" lets exactly one insert win. The loser adopts the
        # winner's row rather than 500-ing the request.
        db.rollback()
        winner = db.scalar(
            select(ChapterContent).where(ChapterContent.chapter_id == chapter_id, ChapterContent.scope == "global")
        )
        if winner is None:
            raise
        return winner
    db.refresh(content)
    return content


def _resolve_content_for_user(chapter_id: int, user_id: int, db: Session) -> ChapterContent:
    """Same precedence as `progression._resolve_relevant_content`: this
    user's own latest remediation (scope="user") version if one exists,
    else the chapter's shared scope="global" version — created on first
    open if it doesn't exist yet. Kept as a separate lookup (not a straight
    import of the private helper) because this one must also fall through to
    `_get_or_create_content`'s create-on-first-open behavior, which
    `progression`'s read-only helper deliberately doesn't do."""
    user_content = db.scalar(
        select(ChapterContent)
        .where(ChapterContent.chapter_id == chapter_id, ChapterContent.scope == "user", ChapterContent.user_id == user_id)
        .order_by(ChapterContent.version.desc())
        .limit(1)
    )
    if user_content is not None:
        return user_content
    return _get_or_create_content(chapter_id, db)


def stream_chapter_content(chapter: Chapter, db: Session, user_id: int) -> Iterator[dict]:
    content = _resolve_content_for_user(chapter.id, user_id, db)

    existing = (
        db.query(ChapterContentSection)
        .filter_by(chapter_content_id=content.id)
        .order_by(ChapterContentSection.order)
        .all()
    )
    for section in existing:
        yield _section_event(section)

    if content.status == "ready":
        yield {"type": "done"}
        return

    if content.scope == "user":
        # A remediation (V2+) version is authored exclusively by the
        # background `remediate_chapter_task` (dispatched at grading time,
        # see app.agents.evaluation.grade), using the narrow
        # generate_remediation_outline — never inline here. Generating it
        # inline with this file's own generate_section_outline would author
        # a full chapter re-teach instead of the targeted few sections the
        # learner actually needs, and would race the background task to
        # write the same row. So: report status and let the caller re-poll
        # instead of generating anything.
        if content.status == "failed" and content.remediation_source_attempt_id is not None:
            # Self-heal instead of leaving the learner permanently stuck on a
            # transient failure: re-dispatch the same row (remediate_chapter
            # only treats "ready" as a true no-op, so this is safe to repeat
            # on every reopen) and report "generating" so the caller re-polls.
            remediate_chapter_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
                content.chapter_id, user_id, content.remediation_target_tags or [],
                content.remediation_source_attempt_id,
            )
        yield {"type": "generating"}
        yield {"type": "done"}
        return

    if not content.outline:
        try:
            outline = generate_section_outline(chapter.title, chapter.objective)
        except Exception as exc:
            content.status = "failed"
            content.error = str(exc)
            content.updated_at = datetime.now(timezone.utc)
            db.commit()
            yield {"type": "error", "message": str(exc)}
            return
        # A concurrent open of the same never-before-generated chapter can
        # reach this point too; re-read the row before writing so a slower
        # generator never overwrites a faster one's already-persisted outline
        # (this UPDATE has no unique constraint to raise IntegrityError, so
        # the check has to happen here instead of via a commit-time conflict).
        db.refresh(content)
        if not content.outline:
            content.outline = [o.model_dump() for o in outline]
            content.updated_at = datetime.now(timezone.utc)
            db.commit()

    done_orders = {s.order for s in existing}
    for i, entry in enumerate(content.outline):
        if i in done_orders:
            continue
        try:
            result = generate_chapter_section(
                chapter.title, chapter.objective, entry["heading"], entry["objective"], entry["kind"],
            )
        except Exception as exc:
            content.status = "failed"
            content.error = str(exc)
            content.updated_at = datetime.now(timezone.utc)
            db.commit()
            yield {"type": "error", "message": str(exc)}
            return

        diagram_spec = result.diagram_spec
        section = ChapterContentSection(
            chapter_content_id=content.id, order=i, heading=entry["heading"], kind=entry["kind"],
            body_markdown=result.body_markdown,
            examples=[e.model_dump() for e in result.examples],
            diagram_spec=diagram_spec.model_dump() if diagram_spec is not None else None,
            diagram_status="pending" if diagram_spec is not None else None,
        )
        db.add(section)
        try:
            db.commit()
        except IntegrityError:
            # A concurrent open of the same chapter generated and persisted this
            # exact section first (unique (chapter_content_id, order)). Adopt the
            # winner's row — don't re-generate or re-dispatch its diagram — and
            # resume from here.
            db.rollback()
            section = db.scalar(
                select(ChapterContentSection).where(
                    ChapterContentSection.chapter_content_id == content.id,
                    ChapterContentSection.order == i,
                )
            )
            if section is None:
                raise
        else:
            db.refresh(section)
            if diagram_spec is not None:
                render_diagram_task.delay(section.id)  # pyright: ignore[reportFunctionMemberAccess]

        yield _section_event(section)

    # Re-check against the DB before flipping to "ready". Under a concurrent
    # double-open both generators loop over the same outline; this generator may
    # have "adopted" the winner's rows (IntegrityError path above) while the
    # winner is still generating later sections. Flipping ready now would let a
    # later crash of the winner strand the chapter "ready" yet incomplete (and it
    # would then never be regenerated). Only claim ready once every outline order
    # actually has a persisted section.
    persisted_orders = {
        s.order for s in db.scalars(
            select(ChapterContentSection).where(
                ChapterContentSection.chapter_content_id == content.id
            )
        )
    }
    if persisted_orders == set(range(len(content.outline))):
        content.status = "ready"
        content.updated_at = datetime.now(timezone.utc)
        db.commit()
        generate_chapter_assignment_task.delay(content.id)  # pyright: ignore[reportFunctionMemberAccess]
    yield {"type": "done"}
