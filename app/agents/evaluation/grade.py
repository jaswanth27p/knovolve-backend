"""Grades one AssignmentAttempt: mcq/true_false answers are graded in-process
by exact match, then a single structured-output LLM call handles free-text
grading AND a holistic remediation-targeting verdict over the WHOLE attempt
(spec 03) — this call runs even when the attempt has zero free-text
questions, since the verdict is needed regardless. For a chapter-level
attempt (Trigger A) with non-empty remediation_concept_tags, this dispatches
narrow remediation content generation for exactly those concepts. Module-
level assignments (Trigger B) never dispatch remediation — that trigger path
is deliberately absent, per spec 03's simplification.

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
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.chapter_content import ChapterContent
from app.agents.evaluation.nodes.grade_assignment_answers import (
    FreeTextAnswerItem, KnownAnswerItem, grade_assignment_answers,
)
from app.services import streaks
from app.tasks.chapter_content_tasks import remediate_chapter_task


def _normalize(text: str) -> str:
    return text.strip().casefold()


def grade_assignment_attempt(attempt_id: int, db: Session) -> None:
    attempt = db.get(AssignmentAttempt, attempt_id)
    if attempt is None:
        raise ValueError(f"AssignmentAttempt {attempt_id} not found")
    if attempt.status != "grading":
        return  # already graded/failed, or a duplicate task delivery

    assignment = db.get(Assignment, attempt.assignment_id)
    if assignment is None:
        raise ValueError(f"Assignment {attempt.assignment_id} not found")

    answers = db.scalars(
        select(AssignmentAnswer).where(AssignmentAnswer.attempt_id == attempt_id)
    ).all()
    question_ids = [a.question_id for a in answers]
    questions = {
        q.id: q for q in db.scalars(
            select(AssignmentQuestion).where(AssignmentQuestion.id.in_(question_ids))
        ).all()
    }

    remediation_concept_tags: list[str] = []
    try:
        free_text_items: list[FreeTextAnswerItem] = []
        known_answers: list[KnownAnswerItem] = []
        for answer in answers:
            question = questions[answer.question_id]
            if question.type in ("mcq", "true_false"):
                is_correct = _normalize(answer.user_answer) == _normalize(question.correct_answer)
                answer.is_correct = is_correct
                answer.feedback = (
                    "Correct." if is_correct
                    else f"Incorrect. The correct answer is: {question.correct_answer}"
                )
                answer.graded_at = datetime.now(timezone.utc)
                known_answers.append(KnownAnswerItem(
                    question_id=question.id, question_text=question.text, concept_tag=question.concept_tag,
                    user_answer=answer.user_answer, is_correct=is_correct,
                ))
            else:
                free_text_items.append(FreeTextAnswerItem(
                    question_id=question.id, question_text=question.text,
                    correct_answer=question.correct_answer, explanation=question.explanation,
                    concept_tag=question.concept_tag, user_answer=answer.user_answer,
                ))

        free_text_question_ids = {i.question_id for i in free_text_items}
        grading = grade_assignment_answers(free_text_items, known_answers)
        grades_by_question = {g.question_id: g for g in grading.grades}
        missing = [i.question_id for i in free_text_items if i.question_id not in grades_by_question]
        if missing:
            raise ValueError(f"grader returned no grade for question_id(s): {missing}")
        for answer in answers:
            # Defense-in-depth: only ever apply an LLM grade to a question
            # that was actually sent as a free-text item. known_answers
            # (mcq/true_false) are context-only for the LLM — if it ever
            # hallucinated a grades[] entry for one of their question_ids,
            # this guard stops that from silently overwriting the
            # deterministic exact-match result already set above.
            if answer.question_id not in free_text_question_ids:
                continue
            grade = grades_by_question.get(answer.question_id)
            if grade is not None:
                answer.is_correct = grade.is_correct
                answer.feedback = grade.feedback
                answer.misconception_tag = grade.misconception_tag
                answer.graded_at = datetime.now(timezone.utc)

        correct_count = sum(1 for a in answers if a.is_correct)
        attempt.overall_score = correct_count / len(answers) if answers else 0.0
        attempt.verdict_reasoning = grading.verdict_reasoning
        attempt.status = "graded"
        attempt.updated_at = datetime.now(timezone.utc)
        streaks.record_activity(db, attempt.user_id, attempt.updated_at)
        # Defense-in-depth, mirroring the grades[]/question_id guard above:
        # only ever persist/dispatch remediation tags that were actually
        # among this attempt's questions' concept_tag values. Without this,
        # a paraphrased or invented tag from the LLM's free-form verdict
        # would get written verbatim into ChapterContent.remediation_target_tags
        # — the durable, permanent weakness record later specs match on by
        # tag — and drive which sections get generated.
        allowed_tags = {q.concept_tag for q in questions.values()}
        remediation_concept_tags = [t for t in grading.remediation_concept_tags if t in allowed_tags]
        db.commit()
    except Exception as exc:
        # Rolls back any in-session, uncommitted grading (e.g. mcq/true_false
        # answers graded before the LLM call failed) — the whole attempt
        # fails, not a partial grade.
        db.rollback()
        attempt.status = "failed"
        attempt.error = str(exc)
        attempt.updated_at = datetime.now(timezone.utc)
        db.commit()
        return

    if assignment.level == "chapter" and remediation_concept_tags:
        content = db.get(ChapterContent, assignment.chapter_content_id)
        if content is not None:
            remediate_chapter_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
                content.chapter_id, attempt.user_id, remediation_concept_tags, attempt.id,
            )
