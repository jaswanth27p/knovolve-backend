"""Grades one AssignmentAttempt: mcq/true_false answers are graded in-process
by exact match, then every free_text answer in the attempt is graded in a
SINGLE batched structured-output LLM call. Plain function, not a LangGraph
graph — same rationale as chapter-assignment generation
(app/agents/assignment/generate.py): a handful of fast calls, nothing
long-running, no async sub-step.

No stale-row recovery/CAS logic here, unlike
`app/agents/assignment/generate.py`'s `_reactivate_if_retryable` — see the
Global Constraints in the plan this was built from for why: attempts aren't
a scarce shared resource (no unique-per-assignment index), so a stuck
"grading" row doesn't block the learner (they just submit a new attempt),
and only this task's own dispatcher (the submit route) ever triggers
grading — there's no cross-endpoint race like the assignment routes'
get-or-create-and-dispatch pattern to defend against.
"""

from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.attempt import AssignmentAttempt, AssignmentAnswer
from app.models.assignment import AssignmentQuestion
from app.agents.evaluation.nodes.grade_free_text_answers import FreeTextAnswerItem, grade_free_text_answers
from app.services import streaks


def _normalize(text: str) -> str:
    return text.strip().casefold()


def grade_assignment_attempt(attempt_id: int, db: Session) -> None:
    attempt = db.get(AssignmentAttempt, attempt_id)
    if attempt is None:
        raise ValueError(f"AssignmentAttempt {attempt_id} not found")
    if attempt.status != "grading":
        return  # already graded/failed, or a duplicate task delivery

    answers = db.scalars(
        select(AssignmentAnswer).where(AssignmentAnswer.attempt_id == attempt_id)
    ).all()
    question_ids = [a.question_id for a in answers]
    questions = {
        q.id: q for q in db.scalars(
            select(AssignmentQuestion).where(AssignmentQuestion.id.in_(question_ids))
        ).all()
    }

    try:
        free_text_items: list[FreeTextAnswerItem] = []
        for answer in answers:
            question = questions[answer.question_id]
            if question.type in ("mcq", "true_false"):
                answer.is_correct = _normalize(answer.user_answer) == _normalize(question.correct_answer)
                answer.feedback = (
                    "Correct." if answer.is_correct
                    else f"Incorrect. The correct answer is: {question.correct_answer}"
                )
                answer.graded_at = datetime.now(timezone.utc)
            else:
                free_text_items.append(FreeTextAnswerItem(
                    question_id=question.id, question_text=question.text,
                    correct_answer=question.correct_answer, explanation=question.explanation,
                    user_answer=answer.user_answer,
                ))

        if free_text_items:
            grades_by_question = {g.question_id: g for g in grade_free_text_answers(free_text_items)}
            missing = [i.question_id for i in free_text_items if i.question_id not in grades_by_question]
            if missing:
                raise ValueError(f"grader returned no grade for question_id(s): {missing}")
            for answer in answers:
                grade = grades_by_question.get(answer.question_id)
                if grade is not None:
                    answer.is_correct = grade.is_correct
                    answer.feedback = grade.feedback
                    answer.graded_at = datetime.now(timezone.utc)

        correct_count = sum(1 for a in answers if a.is_correct)
        attempt.overall_score = correct_count / len(answers) if answers else 0.0
        attempt.status = "graded"
        attempt.updated_at = datetime.now(timezone.utc)
        streaks.record_activity(db, attempt.user_id, attempt.updated_at)
        db.commit()
    except Exception as exc:
        # Rolls back any in-session, uncommitted grading (e.g. mcq/true_false
        # answers graded before a free_text batch call failed) — the whole
        # attempt fails, not a partial grade.
        db.rollback()
        attempt.status = "failed"
        attempt.error = str(exc)
        attempt.updated_at = datetime.now(timezone.utc)
        db.commit()
