from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from app.models.course import Course
from app.models.enrollment import UserCourse


def touch_enrollment(db: Session, user_id: int, course: Course) -> None:
    """Idempotently mark `course` as tracked by `user_id`: upsert on
    (user_id, course_id) so concurrent opens never race the unique constraint
    — insert on first engagement, else bump `last_opened_at`. Never changes
    status/progress (those are owned by the deferred progression engine)."""
    now = datetime.now(timezone.utc)
    stmt = pg_insert(UserCourse).values(
        user_id=user_id, course_id=course.id,
        enrolled_at=now, last_opened_at=now,
    ).on_conflict_do_update(
        index_elements=["user_id", "course_id"],
        set_={"last_opened_at": now},
    )
    db.execute(stmt)
    db.commit()
