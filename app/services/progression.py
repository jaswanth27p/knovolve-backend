"""Course-level progress/status, recomputed after every graded attempt.
"Passed" is a coarser pass/fail gate than mastery's weak/developing/strong
bands (app.services.mastery) — different question (is this CHAPTER done)
from mastery's (is this CONCEPT understood). See design doc §6."""
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Module
from app.models.enrollment import UserCourse

PASS_THRESHOLD = 0.7


def resolve_course_id(db: Session, assignment: Assignment) -> int | None:
    """Resolves the course_id an assignment belongs to, via explicit FK
    lookups — there are no ORM relationship() declarations on Course/Module/
    Chapter/Assignment to traverse."""
    if assignment.level == "chapter":
        if assignment.chapter_content_id is None:
            return None
        content = db.get(ChapterContent, assignment.chapter_content_id)
        if content is None:
            return None
        chapter = db.get(Chapter, content.chapter_id)
        if chapter is None:
            return None
        module = db.get(Module, chapter.module_id)
        if module is None:
            return None
        return module.course_id
    if assignment.level == "module":
        if assignment.module_id is None:
            return None
        module = db.get(Module, assignment.module_id)
        if module is None:
            return None
        return module.course_id
    return None


def _has_passing_attempt(db: Session, assignment_id: int, user_id: int) -> bool:
    """True if ANY graded attempt on this assignment cleared PASS_THRESHOLD,
    not merely the latest one.

    Design doc 05 makes ``chapter_passed`` permanent once true, and spec 02's
    weak/strong detection depends on that invariant: a learner can submit a
    new, worse attempt on an already-passed chapter (the attempts endpoint
    allows retakes), and that must not un-pass the chapter — otherwise a
    cleared concept would flicker back to "weak" and the course's
    ``status='completed'`` would disagree with a recomputed progress < 1.
    """
    return db.scalar(
        select(AssignmentAttempt.id)
        .where(
            AssignmentAttempt.assignment_id == assignment_id,
            AssignmentAttempt.user_id == user_id,
            AssignmentAttempt.status == "graded",
            AssignmentAttempt.overall_score >= PASS_THRESHOLD,
        )
        .limit(1)
    ) is not None


def _resolve_relevant_content(db: Session, chapter_id: int, user_id: int) -> ChapterContent | None:
    """Which ChapterContent version currently applies to this user for this
    chapter: their own latest scope="user" remediation version if one
    exists, else the chapter's scope="global" version."""
    user_content = db.scalar(
        select(ChapterContent)
        .where(
            ChapterContent.chapter_id == chapter_id,
            ChapterContent.scope == "user",
            ChapterContent.user_id == user_id,
        )
        .order_by(ChapterContent.version.desc())
        .limit(1)
    )
    if user_content is not None:
        return user_content
    return db.scalar(
        select(ChapterContent).where(ChapterContent.chapter_id == chapter_id, ChapterContent.scope == "global")
    )


def _chapter_passed(db: Session, chapter_id: int, user_id: int) -> bool:
    content = _resolve_relevant_content(db, chapter_id, user_id)
    if content is None:
        return False
    assignment = db.scalar(
        select(Assignment).where(
            Assignment.level == "chapter",
            Assignment.scope == content.scope,
            Assignment.chapter_content_id == content.id,
            *([Assignment.user_id == user_id] if content.scope == "user" else []),
        )
    )
    if assignment is None:
        return False
    return _has_passing_attempt(db, assignment.id, user_id)


def passed_chapter_ids(db: Session, user_id: int, chapter_ids: list[int]) -> set[int]:
    """Batched equivalent of :func:`_chapter_passed` over many chapters.

    Hot read paths (dashboard serialization, chat context) would otherwise
    issue several queries per chapter. Resolves each chapter's currently
    relevant content (latest user-scoped version, else global), its chapter
    assignment, and whether any of the learner's graded attempts on that
    assignment cleared the threshold — in a constant number of queries."""
    if not chapter_ids:
        return set()
    contents = db.scalars(
        select(ChapterContent).where(
            ChapterContent.chapter_id.in_(chapter_ids),
            or_(
                ChapterContent.scope == "global",
                and_(ChapterContent.scope == "user", ChapterContent.user_id == user_id),
            ),
        )
    ).all()
    chosen: dict[int, ChapterContent] = {}
    for content in contents:
        current = chosen.get(content.chapter_id)
        if current is None or (content.scope, content.version) > (current.scope, current.version):
            chosen[content.chapter_id] = content
    if not chosen:
        return set()
    content_ids = [c.id for c in chosen.values()]
    assignments = db.scalars(
        select(Assignment).where(
            Assignment.level == "chapter",
            Assignment.chapter_content_id.in_(content_ids),
            or_(Assignment.scope == "global", Assignment.user_id == user_id),
        )
    ).all()
    # Map content -> assignment, keeping only the assignment matching the
    # content's scope (a user-scoped content pairs with a user-scoped
    # assignment for this user; a global content with a global assignment).
    by_content = {content.id: content for content in chosen.values()}
    assignment_by_content: dict[int, Assignment] = {}
    for assignment in assignments:
        content_id = assignment.chapter_content_id
        if content_id is None:
            continue
        content = by_content.get(content_id)
        if content is None:
            continue
        if assignment.scope == content.scope and (
            content.scope == "global" or assignment.user_id == user_id
        ):
            assignment_by_content[content_id] = assignment
    assignment_ids = [a.id for a in assignment_by_content.values()]
    if not assignment_ids:
        return set()
    passing_ids = set(
        db.scalars(
            select(AssignmentAttempt.assignment_id)
            .where(
                AssignmentAttempt.assignment_id.in_(assignment_ids),
                AssignmentAttempt.user_id == user_id,
                AssignmentAttempt.status == "graded",
                AssignmentAttempt.overall_score >= PASS_THRESHOLD,
            )
            .distinct()
        ).all()
    )
    return {
        chapter_id
        for chapter_id, content in chosen.items()
        if (a := assignment_by_content.get(content.id)) is not None and a.id in passing_ids
    }


def update_course_progress(db: Session, user_id: int, course_id: int) -> None:
    uc = db.scalar(select(UserCourse).where(UserCourse.user_id == user_id, UserCourse.course_id == course_id))
    if uc is None:
        return  # no enrollment row yet — nothing to update

    chapter_ids = [
        c.id for c in db.scalars(
            select(Chapter).join(Module, Chapter.module_id == Module.id).where(
                Module.course_id == course_id,
                or_(Module.scope == "global", and_(Module.scope == "user", Module.user_id == user_id)),
            )
        ).all()
    ]
    if not chapter_ids:
        return

    passed = len(passed_chapter_ids(db, user_id, chapter_ids))
    uc.progress = passed / len(chapter_ids)
    if uc.progress >= 1.0:
        uc.status = "completed"
    db.commit()
