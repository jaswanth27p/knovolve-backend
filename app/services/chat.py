"""Floating chatbot: a couple of question shapes (progress, latest-result)
are answered deterministically since the data lookup is cheap and exact —
no reason to pay for an LLM call to report a number we already have. Every
other message is free-form and goes to an LLM grounded in the learner's
actual context (dashboard stats, current course/chapter, weak/strong
concepts, latest attempt) — see CHAT_REPLY_PROMPT. No chat history is
persisted (master spec §4.7): each call is stateless."""
from typing import Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.factory import get_chat_model
from app.llm.prompts import CHAT_REPLY_PROMPT
from app.llm.retry import call_with_retry
from app.models.attempt import AssignmentAttempt
from app.models.course import Chapter, Course
from app.services.assignments import _chapter_assignment
from app.services.mastery import get_concept_statuses
from app.services.progression import PASS_THRESHOLD, _resolve_relevant_content
from app.services.tracking import get_dashboard


def _latest_chapter_attempt(db: Session, chapter_id: int, user_id: int) -> AssignmentAttempt | None:
    content = _resolve_relevant_content(db, chapter_id, user_id)
    if content is None:
        return None
    assignment = _chapter_assignment(db, content)
    if assignment is None:
        return None
    return db.scalar(
        select(AssignmentAttempt)
        .where(
            AssignmentAttempt.assignment_id == assignment.id,
            AssignmentAttempt.user_id == user_id,
            AssignmentAttempt.status == "graded",
        )
        .order_by(AssignmentAttempt.created_at.desc())
        .limit(1)
    )


def _handle_progress_question(db: Session, user_id: int) -> str:
    dashboard = get_dashboard(db, user_id)
    return f"You have {dashboard.in_progress_count} course(s) in progress and {dashboard.completed_count} completed."


def _handle_result_question(db: Session, chapter_id: int | None, user_id: int) -> str:
    if chapter_id is None:
        return "Please open a chapter first so I know which assignment result you mean."
    attempt = _latest_chapter_attempt(db, chapter_id, user_id)
    if attempt is None or attempt.overall_score is None:
        return "You haven't completed this chapter's assignment yet."
    pct = round(attempt.overall_score * 100)
    verdict = "you passed" if attempt.overall_score >= PASS_THRESHOLD else "not a pass yet"
    return f"Your latest attempt on this chapter scored {pct}% — {verdict}."


def _build_context(db: Session, user_id: int, course_slug: str | None, chapter_id: int | None) -> str:
    dashboard = get_dashboard(db, user_id)
    lines = [
        f"{dashboard.in_progress_count} course(s) in progress, {dashboard.completed_count} completed, "
        f"current streak {dashboard.streak.current} day(s).",
    ]

    course = db.scalar(select(Course).where(Course.topic_slug == course_slug)) if course_slug else None
    if course is not None:
        lines.append(f'Currently viewing course: "{course.topic_raw}".')
        statuses = get_concept_statuses(db, user_id, course.id)
        weak = sorted(tag for tag, status in statuses.items() if status == "weak")
        strong = sorted(tag for tag, status in statuses.items() if status == "strong")
        if weak:
            lines.append(f"Weak concepts: {', '.join(weak)}.")
        if strong:
            lines.append(f"Strong concepts: {', '.join(strong)}.")

    if chapter_id is not None:
        chapter = db.get(Chapter, chapter_id)
        if chapter is not None:
            lines.append(f'Currently viewing chapter: "{chapter.title}" (objective: {chapter.objective}).')
        attempt = _latest_chapter_attempt(db, chapter_id, user_id)
        if attempt is not None and attempt.overall_score is not None:
            pct = round(attempt.overall_score * 100)
            verdict = "passed" if attempt.overall_score >= PASS_THRESHOLD else "not passed"
            lines.append(f"Latest assignment attempt on this chapter: {pct}% ({verdict}).")

    return "\n".join(lines)


def _handle_freeform_question(db: Session, user_id: int, course_slug: str | None,
                              chapter_id: int | None, message: str) -> str:
    context = _build_context(db, user_id, course_slug, chapter_id)
    model = get_chat_model("chat_reply")
    messages = CHAT_REPLY_PROMPT.format_messages(context=context, message=message)
    resp = call_with_retry(model.invoke, messages)
    content = resp.content
    if not isinstance(content, str):
        raise TypeError(f"expected str content from LLM response, got {type(content)}")
    return content.strip()


def answer_chat_message(db: Session, user_id: int, course_slug: str | None, chapter_id: int | None, message: str) -> str:
    lowered = message.lower()
    if "progress" in lowered or "weak" in lowered or "strong" in lowered:
        return _handle_progress_question(db, user_id)
    if "wrong" in lowered or "result" in lowered or "score" in lowered:
        return _handle_result_question(db, chapter_id, user_id)
    return _handle_freeform_question(db, user_id, course_slug, chapter_id, message)


def stream_chat_message(
    db: Session, user_id: int, course_slug: str | None, chapter_id: int | None, message: str,
) -> Iterator[dict]:
    """Same routing as answer_chat_message, but yields NDJSON-ready events as
    they become available: the deterministic replies resolve instantly and
    stream as a single token, while the free-form path streams the LLM's
    output token-by-token so the UI can render it as it arrives instead of
    waiting for the full reply."""
    lowered = message.lower()
    if "progress" in lowered or "weak" in lowered or "strong" in lowered:
        yield {"type": "token", "text": _handle_progress_question(db, user_id)}
        yield {"type": "done"}
        return
    if "wrong" in lowered or "result" in lowered or "score" in lowered:
        yield {"type": "token", "text": _handle_result_question(db, chapter_id, user_id)}
        yield {"type": "done"}
        return

    context = _build_context(db, user_id, course_slug, chapter_id)
    model = get_chat_model("chat_reply")
    messages = CHAT_REPLY_PROMPT.format_messages(context=context, message=message)
    try:
        for chunk in model.stream(messages):
            content = chunk.content
            if isinstance(content, str) and content:
                yield {"type": "token", "text": content}
    except Exception:  # noqa: BLE001 - surfaced to the client as a chat error, not a 500
        yield {"type": "error", "message": "Failed to generate a reply. Please try again."}
        return
    yield {"type": "done"}
