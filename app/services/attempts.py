"""Attempt service: validate + persist assignment attempt submissions, fetch
graded results, and dispatch async grading."""
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.assignment import AssignmentQuestion
from app.models.attempt import AssignmentAnswer, AssignmentAttempt
from app.models.course import Course
from app.schemas.attempt import (
    AnswerResult,
    AttemptResponse,
    AttemptSubmitResponse,
    ConceptScore,
    SubmitAttemptRequest,
)
from app.services import assignments
from app.tasks.evaluation_tasks import grade_assignment_attempt_task


def submit_attempt(db: Session, user_id: int, course: Course, assignment_id: int,
                   body: SubmitAttemptRequest) -> AttemptSubmitResponse:
    """Validate and persist a new attempt for `assignment_id` in `course`,
    then dispatch async grading. Looks up the assignment via
    `assignments.get_assignment_for_course` (404s "assignment not found" for
    missing/wrong-course), then adds its OWN "ready" status gate on top —
    that shared lookup deliberately skips status so `get_attempt` below can
    reuse it without gaining a status check it never had. Together the two
    checks reproduce the old combined
    `assignment.status != "ready" or not _assignment_belongs_to_course(...)`
    condition's exact set of 404 outcomes."""
    assignment = assignments.get_assignment_for_course(db, assignment_id, course)
    if assignment.status != "ready":
        raise HTTPException(status_code=404, detail="assignment not found")

    questions = db.scalars(
        select(AssignmentQuestion).where(
            AssignmentQuestion.assignment_id == assignment.id,
            (AssignmentQuestion.user_id.is_(None)) | (AssignmentQuestion.user_id == user_id),
        )
    ).all()
    questions_by_id = {q.id: q for q in questions}
    submitted_ids = {a.question_id for a in body.answers}
    if len(body.answers) != len(submitted_ids):
        raise HTTPException(status_code=400, detail="duplicate question_id in submitted answers")
    if submitted_ids != set(questions_by_id.keys()):
        raise HTTPException(status_code=400, detail="submitted answers must cover exactly the assignment's questions")

    now = datetime.now(timezone.utc)
    attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user_id, status="grading",
                                 created_at=now, updated_at=now)
    db.add(attempt)
    db.flush()
    for a in body.answers:
        question = questions_by_id[a.question_id]
        db.add(AssignmentAnswer(attempt_id=attempt.id, question_id=question.id,
                                 concept_tag=question.concept_tag, user_answer=a.answer))
    db.commit()

    grade_assignment_attempt_task.delay(attempt.id)  # pyright: ignore[reportFunctionMemberAccess]
    return AttemptSubmitResponse(attempt_id=attempt.id, status="grading")


def _serialize_attempt(attempt: AssignmentAttempt, db: Session, user_id: int) -> AttemptResponse:
    """`user_id` scopes the concept-score breakdown to questions this exact
    user owns (base questions plus their own module topup, never another
    learner's) — safe because every caller has already verified
    `attempt.user_id == user_id` before calling this."""
    if attempt.status != "graded":
        return AttemptResponse(status=attempt.status, error=attempt.error)
    answers = db.scalars(
        select(AssignmentAnswer)
        .join(AssignmentQuestion, AssignmentQuestion.id == AssignmentAnswer.question_id)
        .where(
            AssignmentAnswer.attempt_id == attempt.id,
            (AssignmentQuestion.user_id.is_(None)) | (AssignmentQuestion.user_id == user_id),
        )
    ).all()
    concept_totals: dict[str, list[int]] = {}
    for a in answers:
        totals = concept_totals.setdefault(a.concept_tag, [0, 0])
        totals[1] += 1
        if a.is_correct:
            totals[0] += 1
    return AttemptResponse(
        status="graded",
        overall_score=attempt.overall_score,
        answers=[
            AnswerResult(question_id=a.question_id, is_correct=bool(a.is_correct), feedback=a.feedback or "")
            for a in answers
        ],
        concept_scores=[
            ConceptScore(concept_tag=tag, correct=c, total=t) for tag, (c, t) in concept_totals.items()
        ],
    )


def get_attempt(db: Session, user_id: int, course: Course, assignment_id: int,
                attempt_id: int) -> AttemptResponse:
    """Fetch attempt `attempt_id` for `assignment_id` in `course`, verifying
    it belongs to both that assignment and `user_id`. The assignment lookup
    via `assignments.get_assignment_for_course` does NOT gate on status
    (unlike `submit_attempt` above) — the current fetch endpoint never
    checked status either, so no extra gate is added here."""
    assignment = assignments.get_assignment_for_course(db, assignment_id, course)
    attempt = db.get(AssignmentAttempt, attempt_id)
    if attempt is None or attempt.assignment_id != assignment.id or attempt.user_id != user_id:
        raise HTTPException(status_code=404, detail="attempt not found")
    return _serialize_attempt(attempt, db, user_id)
