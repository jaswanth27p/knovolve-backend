from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.course import Course
from app.models.enrollment import UserCourse


def touch_enrollment(db: Session, user_id: int, course: Course) -> None:
    """Idempotently mark `course` as tracked by `user_id`: insert on first
    engagement, else bump `last_opened_at`. Never changes status/progress
    (those are owned by the deferred progression engine)."""
    row = db.scalar(
        select(UserCourse).where(
            UserCourse.user_id == user_id, UserCourse.course_id == course.id
        )
    )
    now = datetime.now(timezone.utc)
    if row is None:
        db.add(UserCourse(
            user_id=user_id, course_id=course.id,
            enrolled_at=now, last_opened_at=now,
        ))
    else:
        row.last_opened_at = now
    db.commit()