"""Generates a chapter's assignment: one structured-output LLM call per
teaching section, topped up from a whole-chapter prompt if the total falls
short of MIN_QUESTIONS. Plain function, not a LangGraph graph — nothing
here is long-running or has an async sub-step (unlike chapter content's
diagram rendering), so a crash just retries the whole thing; there is
nothing slow enough mid-way to need per-step persistence.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.config import settings
from app.models.assignment import Assignment, AssignmentQuestion, AssignmentUserTopup
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter, Module
from app.services import mastery
from app.agents.assignment.nodes.generate_section_questions import generate_questions_for_section
from app.agents.assignment.nodes.generate_topup_questions import generate_topup_questions
from app.agents.assignment.nodes.generate_weak_concept_questions import generate_weak_concept_questions

MIN_QUESTIONS = 3

# How long a row may sit in "generating" before a retry is allowed to take it
# over. It has to exceed the longest a *healthy* run can hold the row, or a
# retry would start generating alongside one that is still working. Celery's
# hard task_time_limit is exactly that bound (the run is killed at it), so the
# limit plus a generous margin for clock skew between the worker that wrote
# updated_at and the one reading it is safe. Deliberately NOT the broker's
# visibility_timeout (6h): that bounds redelivery, not run length, and would
# leave a learner polling a dead assignment for six hours.
STALE_AFTER = timedelta(seconds=settings.celery_task_time_limit_seconds) + timedelta(minutes=30)


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


def _reactivate_if_retryable(db: Session, assignment: Assignment) -> bool:
    """Decide whether this run may (re)generate an Assignment row that already
    existed, and claim it if so. Returns True when the caller owns generation.

    Retryable states are "failed" and a *stale* "generating" — the latter is a
    run whose worker died mid-flight: `task_acks_late` gets the job redelivered,
    but without this the redelivered run would see "generating" and no-op,
    stranding the row (and the polling client) forever.

    The claim is a compare-and-swap on the exact (status, updated_at) pair this
    run observed, so of two concurrent retries only one proceeds. Matching on
    updated_at as well as status is what makes the stale case safe: both racers
    observe status="generating", so status alone would let both CAS through.
    """
    observed_status = assignment.status
    observed_updated_at = assignment.updated_at

    if observed_status == "generating":
        updated_at = observed_updated_at
        if updated_at.tzinfo is None:  # defensive: a naive column read
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        if updated_at > datetime.now(timezone.utc) - STALE_AFTER:
            return False  # a healthy generation is still in flight
    elif observed_status != "failed":
        return False  # "ready" — nothing to do

    rows = db.execute(
        update(Assignment)
        .where(
            Assignment.id == assignment.id,
            Assignment.status == observed_status,
            Assignment.updated_at == observed_updated_at,
        )
        .values(status="generating", error=None, updated_at=datetime.now(timezone.utc))
    ).rowcount
    db.commit()
    if rows == 0:
        return False  # another retry won the reactivation race

    # A prior failed/partial/crashed run may have left question rows behind. A
    # retry regenerates the whole set, so clear them rather than interleaving
    # fresh questions with stale ones (and colliding on the unique order index).
    db.execute(delete(AssignmentQuestion).where(AssignmentQuestion.assignment_id == assignment.id))
    db.commit()
    db.refresh(assignment)
    return True


def _mark_failed(db: Session, assignment: Assignment, exc: Exception) -> None:
    db.rollback()
    assignment.status = "failed"
    assignment.error = str(exc)
    assignment.updated_at = datetime.now(timezone.utc)
    db.commit()


def _persist_questions(db: Session, assignment: Assignment, questions: Iterable[tuple[int | None, Any]]) -> None:
    """Write the generated questions in order and flip the assignment to
    "ready". `questions` is (source_section_id, question-like) pairs — either
    a freshly generated QuestionDraft or an existing AssignmentQuestion being
    reused by a module assignment."""
    for order, (section_id, q) in enumerate(questions):
        db.add(AssignmentQuestion(
            assignment_id=assignment.id, order=order, type=q.type, text=q.text,
            # Copy: a reused AssignmentQuestion's JSONB list would otherwise be
            # the same mutable object on both rows.
            options=list(q.options) if q.options else None,
            correct_answer=q.correct_answer, explanation=q.explanation, concept_tag=q.concept_tag,
            difficulty=q.difficulty, source_section_id=section_id,
        ))
    assignment.status = "ready"
    assignment.updated_at = datetime.now(timezone.utc)
    db.commit()


def _teaching_sections(db: Session, chapter_content_id: int) -> Sequence[ChapterContentSection]:
    return db.scalars(
        select(ChapterContentSection)
        .where(
            ChapterContentSection.chapter_content_id == chapter_content_id,
            ChapterContentSection.kind == "teaching",
        )
        .order_by(ChapterContentSection.order)
    ).all()


def generate_chapter_assignment(chapter_content_id: int, db: Session) -> None:
    content = db.get(ChapterContent, chapter_content_id)
    if content is None:
        raise ValueError(f"ChapterContent {chapter_content_id} not found")
    chapter = db.get(Chapter, content.chapter_id)

    assignment, created = _get_or_create_assignment(
        db, level="chapter", scope=content.scope, chapter_content_id=chapter_content_id,
        user_id=content.user_id if content.scope == "user" else None,
    )
    if not created and not _reactivate_if_retryable(db, assignment):
        return  # already ready, a healthy run is in flight, or we lost the race

    sections = _teaching_sections(db, chapter_content_id)

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

        _persist_questions(
            db, assignment,
            [(section_id, q) for section_id, drafts in drafts_by_section for q in drafts],
        )
    except Exception as exc:
        # Covers persistence too, not just the LLM calls: a duplicate-order
        # IntegrityError here must land as status="failed", not as an
        # unhandled Celery task crash.
        _mark_failed(db, assignment, exc)
        return


def _spread_by_concept(questions: Sequence[AssignmentQuestion], count: int) -> list[AssignmentQuestion]:
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
    if not created and not _reactivate_if_retryable(db, assignment):
        return  # already ready, a healthy run is in flight, or we lost the race

    chapters = db.scalars(
        select(Chapter).where(Chapter.module_id == module_id).order_by(Chapter.order)
    ).all()

    try:
        all_drafts: list[tuple[int | None, Any]] = []
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
            sections = _teaching_sections(db, content.id)

            if chapter_assignment is not None:
                existing_questions = db.scalars(
                    select(AssignmentQuestion)
                    .where(AssignmentQuestion.assignment_id == chapter_assignment.id)
                    .order_by(AssignmentQuestion.order)
                ).all()
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

        _persist_questions(db, assignment, all_drafts)
    except Exception as exc:
        _mark_failed(db, assignment, exc)
        return


def _get_or_create_topup(db: Session, assignment_id: int, user_id: int) -> tuple[AssignmentUserTopup, bool]:
    """Get-or-create against the (assignment_id, user_id) unique constraint.
    Same shape as _get_or_create_assignment — created=False means a row
    already existed (any status) and the caller decides what to do."""
    filters = [AssignmentUserTopup.assignment_id == assignment_id, AssignmentUserTopup.user_id == user_id]
    existing = db.scalar(select(AssignmentUserTopup).where(*filters))
    if existing is not None:
        return existing, False

    now = datetime.now(timezone.utc)
    topup = AssignmentUserTopup(
        assignment_id=assignment_id, user_id=user_id, status="generating", created_at=now, updated_at=now,
    )
    db.add(topup)
    try:
        db.commit()
    except IntegrityError:
        # Two concurrent triggers for the same learner (e.g. a racing double
        # fetch of the module assignment) can both pass the select above; the
        # unique constraint lets exactly one insert win.
        db.rollback()
        winner = db.scalar(select(AssignmentUserTopup).where(*filters))
        if winner is None:
            raise
        return winner, False
    db.refresh(topup)
    return topup, True


def _reactivate_topup_if_retryable(db: Session, topup: AssignmentUserTopup) -> bool:
    """Decide whether this run may (re)generate an AssignmentUserTopup row that
    already existed, and claim it if so. Returns True when the caller owns
    generation. Same CAS shape as _reactivate_if_retryable, scoped to
    AssignmentUserTopup.

    Retryable states are "failed" and a *stale* "generating" — the latter is a
    run whose worker died mid-flight: `task_acks_late` gets the job redelivered,
    but without this the redelivered run would see "generating" and no-op,
    stranding the row (and the polling client) forever. "ready" and "skipped"
    are BOTH terminal — a topup that already determined "nothing to add"
    (skipped) must not be regenerated on a later fetch; that would let a
    learner who later develops a new weak concept never get topped up again
    after their first "skipped" result, which is an acceptable limitation for
    V1 (no product requirement forces re-checking on every fetch), not a bug
    to work around here.

    The claim is a compare-and-swap on the exact (status, updated_at) pair this
    run observed, so of two concurrent retries only one proceeds. Matching on
    updated_at as well as status is what makes the stale case safe: both racers
    observe status="generating", so status alone would let both CAS through.
    """
    observed_status = topup.status
    observed_updated_at = topup.updated_at

    if observed_status == "generating":
        updated_at = observed_updated_at
        if updated_at.tzinfo is None:  # defensive: a naive column read
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        if updated_at > datetime.now(timezone.utc) - STALE_AFTER:
            return False  # a healthy generation is still in flight
    elif observed_status != "failed":
        return False  # "ready" or "skipped" — nothing to do

    rows = db.execute(
        update(AssignmentUserTopup)
        .where(
            AssignmentUserTopup.id == topup.id,
            AssignmentUserTopup.status == observed_status,
            AssignmentUserTopup.updated_at == observed_updated_at,
        )
        .values(status="generating", error=None, updated_at=datetime.now(timezone.utc))
    ).rowcount
    db.commit()
    if rows == 0:
        return False  # another retry won the reactivation race

    # A prior failed/crashed run may have left this user's topup questions
    # behind. A retry regenerates the whole topup set, so clear them rather
    # than interleaving fresh questions with stale ones (and colliding on the
    # unique order index). This delete can itself fail (e.g. a learner already
    # submitted an attempt whose assignment_answers.question_id FKs one of
    # these rows) — the CAS above already flipped the row to "generating", so
    # an uncaught failure here would strand it there forever. Mirror
    # generate_module_topup's own except block: mark the row "failed" instead.
    try:
        db.execute(
            delete(AssignmentQuestion).where(
                AssignmentQuestion.assignment_id == topup.assignment_id,
                AssignmentQuestion.user_id == topup.user_id,
            )
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        topup.status = "failed"
        topup.error = str(exc)
        topup.updated_at = datetime.now(timezone.utc)
        db.commit()
        return False
    db.refresh(topup)
    return True


TOPUP_QUESTION_COUNT = 4  # deliberate fixed count, not derived from len(weak_tags) — see design note


def _insert_topup_questions(db: Session, assignment_id: int, user_id: int, drafts: Sequence[Any]) -> None:
    """Insert `drafts` as AssignmentQuestion rows continuing from the shared
    assignment's current max order, retrying once on an order collision.

    Two learners' topups can generate concurrently against the same shared
    module assignment and both read the same max_order before either
    commits; the loser's insert then raises IntegrityError on
    `ix_assignment_questions_unique_order` (unique on (assignment_id, order),
    with no user_id in it). A single retry re-reads max_order fresh and
    re-inserts with new order values — sufficient for this scale; a full
    retry loop/backoff would be over-engineering for V1."""
    for attempt in range(2):
        max_order = db.scalar(
            select(func.max(AssignmentQuestion.order)).where(AssignmentQuestion.assignment_id == assignment_id)
        ) or 0
        for i, draft in enumerate(drafts, start=1):
            db.add(AssignmentQuestion(
                assignment_id=assignment_id, user_id=user_id, order=max_order + i,
                type=draft.type, text=draft.text, options=draft.options,
                correct_answer=draft.correct_answer, explanation=draft.explanation,
                concept_tag=draft.concept_tag, difficulty=draft.difficulty, source_section_id=None,
            ))
        try:
            db.commit()
            return
        except IntegrityError:
            db.rollback()
            if attempt == 1:
                raise


def generate_module_topup(assignment_id: int, user_id: int, db: Session) -> None:
    """Per-learner extension of a shared module assignment: adds up to
    TOPUP_QUESTION_COUNT questions targeting THIS user's weak concepts
    (app.services.mastery.get_weak_concept_tags). Never touches the shared
    base question set — every row this creates has user_id=user_id."""
    topup, created = _get_or_create_topup(db, assignment_id, user_id)
    if not created and not _reactivate_topup_if_retryable(db, topup):
        return  # already ready/skipped, a healthy run is in flight, or we lost the race

    try:
        # Lookups (and their failure handling) live inside the try block, not
        # above it: an assignment/module that vanished out from under a
        # retried run must mark the topup "failed", not strand it in
        # "generating" via an uncaught exception (plain `assert` is also
        # stripped under `python -O`, so it's not a safe guard here either).
        assignment = db.get(Assignment, assignment_id)
        if assignment is None or assignment.module_id is None:
            raise ValueError(f"Assignment {assignment_id} not found or not a module assignment")
        module = db.get(Module, assignment.module_id)
        if module is None:
            raise ValueError(f"Module {assignment.module_id} not found")

        weak_tags = mastery.get_weak_concept_tags(db, user_id, module.course_id)
        if not weak_tags:
            topup.status = "skipped"
            topup.updated_at = datetime.now(timezone.utc)
            db.commit()
            return

        chapters = db.scalars(select(Chapter).where(Chapter.module_id == module.id).order_by(Chapter.order)).all()
        all_sections: list[dict] = []
        for chapter in chapters:
            content = db.scalar(
                select(ChapterContent).where(
                    ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global",
                )
            )
            if content is None or content.status != "ready":
                continue
            for section in _teaching_sections(db, content.id):
                all_sections.append({"heading": section.heading, "body_markdown": section.body_markdown})

        drafts = generate_weak_concept_questions(
            module.title, module.objective, all_sections, sorted(weak_tags), TOPUP_QUESTION_COUNT,
        )

        _insert_topup_questions(db, assignment_id, user_id, drafts)
        topup.status = "ready"
        topup.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        db.rollback()
        topup.status = "failed"
        topup.error = str(exc)
        topup.updated_at = datetime.now(timezone.utc)
        db.commit()
