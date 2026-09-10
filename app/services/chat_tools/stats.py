from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.attempt import AssignmentAttempt
from app.services.tracking import get_dashboard


def get_user_stats(db: Session, user_id: int) -> dict:
    dashboard = get_dashboard(db, user_id)
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_count = db.query(AssignmentAttempt).filter(
        AssignmentAttempt.user_id == user_id,
        AssignmentAttempt.created_at >= today_start,
    ).count()
    return {
        "in_progress_count": dashboard.in_progress_count,
        "completed_count": dashboard.completed_count,
        "streak_current": dashboard.streak.current,
        "streak_longest": dashboard.streak.longest,
        "assignments_attempted_today": today_count,
    }
