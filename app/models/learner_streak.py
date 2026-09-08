from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class LearnerStreak(Base):
    __tablename__ = "learner_streaks"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
