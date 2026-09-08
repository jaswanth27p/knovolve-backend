from datetime import datetime, timedelta, timezone
from app.db import SessionLocal
from app.models.user import User
from app.models.learner_streak import LearnerStreak
from app.services.streaks import record_activity


def _make_user(db, email: str) -> int:
    user = User(email=email, password_hash="x")
    db.add(user)
    db.flush()
    return user.id


def test_first_activity_creates_streak_of_one():
    with SessionLocal() as db:
        user_id = _make_user(db, "streak-a@example.com")
        now = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
        record_activity(db, user_id, now)
        db.commit()

    with SessionLocal() as db:
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 1
        assert row.longest_streak == 1
        assert row.last_active_at == now


def test_second_activity_same_utc_day_no_streak_change():
    with SessionLocal() as db:
        user_id = _make_user(db, "streak-b@example.com")
        first = datetime(2026, 9, 8, 8, 0, tzinfo=timezone.utc)
        record_activity(db, user_id, first)
        db.commit()

    with SessionLocal() as db:
        second = datetime(2026, 9, 8, 20, 0, tzinfo=timezone.utc)
        record_activity(db, user_id, second)
        db.commit()

    with SessionLocal() as db:
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 1
        assert row.longest_streak == 1
        assert row.last_active_at == second


def test_activity_next_utc_day_increments_streak():
    with SessionLocal() as db:
        user_id = _make_user(db, "streak-c@example.com")
        day1 = datetime(2026, 9, 8, 23, 0, tzinfo=timezone.utc)
        record_activity(db, user_id, day1)
        db.commit()

    with SessionLocal() as db:
        day2 = datetime(2026, 9, 9, 1, 0, tzinfo=timezone.utc)
        record_activity(db, user_id, day2)
        db.commit()

    with SessionLocal() as db:
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 2
        assert row.longest_streak == 2


def test_gap_of_two_or_more_days_resets_to_one():
    with SessionLocal() as db:
        user_id = _make_user(db, "streak-d@example.com")
        day1 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        record_activity(db, user_id, day1)
        db.commit()
        day2 = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
        record_activity(db, user_id, day2)
        db.commit()
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 2

    with SessionLocal() as db:
        day5 = datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc)  # gap of 3 days
        record_activity(db, user_id, day5)
        db.commit()

    with SessionLocal() as db:
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 1
        assert row.longest_streak == 2  # longest survives the reset


def test_longest_streak_tracks_max_across_multiple_cycles():
    with SessionLocal() as db:
        user_id = _make_user(db, "streak-e@example.com")
        for day in (1, 2, 3):  # streak reaches 3
            record_activity(db, user_id, datetime(2026, 9, day, 10, 0, tzinfo=timezone.utc))
            db.commit()

    with SessionLocal() as db:
        record_activity(db, user_id, datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc))  # gap resets to 1
        db.commit()

    with SessionLocal() as db:
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 1
        assert row.longest_streak == 3


def test_utc_boundary_not_local_boundary():
    """23:59 UTC then 00:01 UTC next day = a genuine UTC-day gap of 1, even
    though the wall-clock instants are 2 minutes apart."""
    with SessionLocal() as db:
        user_id = _make_user(db, "streak-f@example.com")
        record_activity(db, user_id, datetime(2026, 9, 8, 23, 59, tzinfo=timezone.utc))
        db.commit()

    with SessionLocal() as db:
        record_activity(db, user_id, datetime(2026, 9, 9, 0, 1, tzinfo=timezone.utc))
        db.commit()

    with SessionLocal() as db:
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 2


def test_stale_out_of_order_activity_is_ignored():
    """A late-delivered grading event for an earlier UTC day than the stored
    last_active_at must not rewind current_streak or last_active_at."""
    with SessionLocal() as db:
        user_id = _make_user(db, "streak-g@example.com")
        record_activity(db, user_id, datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc))
        db.commit()
        record_activity(db, user_id, datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc))
        db.commit()

    with SessionLocal() as db:
        stale = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)
        record_activity(db, user_id, stale)
        db.commit()

    with SessionLocal() as db:
        row = db.query(LearnerStreak).filter_by(user_id=user_id).one()
        assert row.current_streak == 2
        assert row.last_active_at == datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)
