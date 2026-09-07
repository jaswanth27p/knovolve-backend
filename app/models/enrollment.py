from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class UserCourse(Base):
    __tablename__ = "user_courses"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="in_progress")  # in_progress|completed (completed deferred)
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # reserved for deferred progression engine
    enrolled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("user_id", "course_id", name="uq_user_courses_user_course"),)
