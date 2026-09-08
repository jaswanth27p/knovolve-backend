"""Streak update rule: one row per user in `learner_streaks`, advanced from
the assignment-grading success path (see `grade_assignment_attempt`) every
time an AssignmentAttempt transitions to status="graded". No commit here —
the caller commits this together with its own row change so the streak
update is atomic with whatever triggered it."""
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.learner_streak import LearnerStreak


def record_activity(db: Session, user_id: int, at: datetime) -> None:
    streak = db.scalars(
        select(LearnerStreak).where(LearnerStreak.user_id == user_id)
    ).one_or_none()

    if streak is None:
        db.add(LearnerStreak(user_id=user_id, current_streak=1, longest_streak=1, last_active_at=at))
        return

    activity_day = at.astimezone(timezone.utc).date()
    prior_day = streak.last_active_at.astimezone(timezone.utc).date()
    day_gap = (activity_day - prior_day).days

    if day_gap < 0:
        # Stale, out-of-order delivery (e.g. a retried grading task) for a
        # UTC day earlier than what's already recorded — ignore entirely,
        # don't rewind progress or last_active_at.
        return
    if day_gap == 1:
        streak.current_streak += 1
    elif day_gap >= 2:
        streak.current_streak = 1
    # day_gap == 0: no streak change, only last_active_at advances below.

    streak.longest_streak = max(streak.longest_streak, streak.current_streak)
    streak.last_active_at = at
