from datetime import datetime, timedelta, timezone
from app.db import SessionLocal
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.auth.security import hash_password, hash_token
from app.tasks.auth_tasks import purge_expired_refresh_tokens


def _make_user(email: str) -> int:
    with SessionLocal() as db:
        user = User(email=email, password_hash=hash_password("x"))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user.id


def _make_token(user_id: int, *, revoked_days_ago: float | None, expires_days_from_now: float) -> int:
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        token = RefreshToken(
            user_id=user_id,
            token_hash=hash_token(f"tok-{user_id}-{revoked_days_ago}-{expires_days_from_now}-{now.timestamp()}"),
            expires_at=now + timedelta(days=expires_days_from_now),
            revoked_at=now - timedelta(days=revoked_days_ago) if revoked_days_ago is not None else None,
            created_at=now,
        )
        db.add(token)
        db.commit()
        db.refresh(token)
        return token.id


def test_purge_deletes_long_revoked_tokens():
    user_id = _make_user("purge-revoked@example.com")
    old_revoked_id = _make_token(user_id, revoked_days_ago=31, expires_days_from_now=-1)

    deleted = purge_expired_refresh_tokens()

    assert deleted == 1
    with SessionLocal() as db:
        assert db.get(RefreshToken, old_revoked_id) is None


def test_purge_deletes_never_used_expired_tokens():
    user_id = _make_user("purge-abandoned@example.com")
    abandoned_id = _make_token(user_id, revoked_days_ago=None, expires_days_from_now=-31)

    deleted = purge_expired_refresh_tokens()

    assert deleted == 1
    with SessionLocal() as db:
        assert db.get(RefreshToken, abandoned_id) is None


def test_purge_keeps_recently_revoked_tokens():
    user_id = _make_user("purge-recent@example.com")
    recent_id = _make_token(user_id, revoked_days_ago=1, expires_days_from_now=-1)

    deleted = purge_expired_refresh_tokens()

    assert deleted == 0
    with SessionLocal() as db:
        assert db.get(RefreshToken, recent_id) is not None


def test_purge_keeps_live_unexpired_tokens():
    user_id = _make_user("purge-live@example.com")
    live_id = _make_token(user_id, revoked_days_ago=None, expires_days_from_now=30)

    deleted = purge_expired_refresh_tokens()

    assert deleted == 0
    with SessionLocal() as db:
        assert db.get(RefreshToken, live_id) is not None
