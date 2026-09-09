"""V1 floating chatbot: stateless per-message, no chat history persisted
(master spec §4.7). Deliberately small v1 tool surface — progress and
last-attempt-result only (design doc §8) — routed by simple keyword
matching rather than a full LLM tool-calling loop, to ship the
progress/results Q&A use case now without building general retrieval
over chapter content."""
from sqlalchemy.orm import Session
from app.services.tracking import get_dashboard


def _handle_progress_question(db: Session, user_id: int) -> str:
    dashboard = get_dashboard(db, user_id)
    return f"You have {dashboard.in_progress_count} course(s) in progress and {dashboard.completed_count} completed."


def _handle_result_question(chapter_id: int | None) -> str:
    if chapter_id is None:
        return "Please open a chapter first so I know which assignment result you mean."
    return "Fetching your latest result for this chapter is not wired up yet in this pass."


def answer_chat_message(db: Session, user_id: int, course_slug: str | None, chapter_id: int | None, message: str) -> str:
    lowered = message.lower()
    if "progress" in lowered or "weak" in lowered or "strong" in lowered:
        return _handle_progress_question(db, user_id)
    if "wrong" in lowered or "result" in lowered or "score" in lowered:
        return _handle_result_question(chapter_id)
    return "I can answer questions about your progress or your latest assignment result."
