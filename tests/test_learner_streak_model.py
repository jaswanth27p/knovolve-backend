from datetime import datetime, timezone
from app.db import SessionLocal
from app.models.user import User
from app.models.learner_streak import LearnerStreak


def test_learner_streak_round_trip():
    user_id = None
    with SessionLocal() as db:
        user = User(email="streak-model@example.com", password_hash="x")
        db.add(user)
        db.flush()
        user_id = user.id
        now = datetime.now(timezone.utc)
        db.add(LearnerStreak(user_id=user_id, current_streak=3, longest_streak=5, last_active_at=now))
        db.commit()

    with SessionLocal() as db:
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 3
        assert row.longest_streak == 5
        assert row.last_active_at is not None


def test_user_id_is_unique():
    from sqlalchemy.exc import IntegrityError
    user_id = None
    now = None
    with SessionLocal() as db:
        user = User(email="streak-unique@example.com", password_hash="x")
        db.add(user)
        db.flush()
        user_id = user.id
        now = datetime.now(timezone.utc)
        db.add(LearnerStreak(user_id=user_id, current_streak=1, longest_streak=1, last_active_at=now))
        db.commit()

    with SessionLocal() as db:
        db.add(LearnerStreak(user_id=user_id, current_streak=1, longest_streak=1, last_active_at=now))
        try:
            db.commit()
            assert False, "expected IntegrityError"
        except IntegrityError:
            db.rollback()
