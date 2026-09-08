import threading
import time
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlalchemy import select
from app.main import app
from app.db import SessionLocal, get_session
from app.models.refresh_token import RefreshToken
from app.auth.security import hash_token

client = TestClient(app, base_url="https://testserver")
CSRF_HEADERS = {"X-Requested-With": "knovolve"}


def _register(email="r@example.com") -> str:
    client.cookies.clear()
    client.post("/auth/register", json={"email": email, "password": "pw123456"}, headers=CSRF_HEADERS)
    refresh_token = client.cookies.get("refresh_token")
    assert refresh_token is not None
    return refresh_token


def test_refresh_rotates_token():
    old_refresh = _register()
    resp = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert resp.status_code == 200
    assert client.cookies.get("refresh_token") != old_refresh


def test_refresh_without_csrf_header_rejected():
    _register("csrf-refresh@example.com")
    resp = client.post("/auth/refresh")
    assert resp.status_code == 403


def test_reused_refresh_token_revokes_chain():
    old_refresh = _register("reuse@example.com")
    with SessionLocal() as db:
        user_id = db.scalar(
            select(RefreshToken.user_id).where(RefreshToken.token_hash == hash_token(old_refresh))
        )
    client.post("/auth/refresh", headers=CSRF_HEADERS)
    # Replay the original (now-revoked) token explicitly, bypassing whatever
    # the jar rotated it to.
    client.cookies.set("refresh_token", old_refresh)
    resp = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert resp.status_code == 401

    with SessionLocal() as db:
        all_tokens = db.query(RefreshToken).filter(RefreshToken.user_id == user_id).all()
        assert all(t.revoked_at is not None for t in all_tokens)


def test_logout_revokes_token():
    _register("logout@example.com")
    client.post("/auth/logout", headers=CSRF_HEADERS)
    resp = client.get("/me/courses", headers=CSRF_HEADERS)
    assert resp.status_code == 401
    assert client.cookies.get("access_token") is None
    assert client.cookies.get("refresh_token") is None


def test_logout_without_csrf_header_rejected():
    _register("logout-csrf@example.com")
    resp = client.post("/auth/logout")
    assert resp.status_code == 403


def test_reuse_of_already_expired_revoked_token_still_revokes_chain():
    """Regression test (round 2): a delayed-replay attack - steal a token,
    wait out its own TTL, then replay it - must still be caught as reuse
    and revoke the whole chain, not short-circuit into a bare "expired" 401
    that skips chain revocation."""
    old_refresh = _register("delayed-replay@example.com")

    first = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert first.status_code == 200

    with SessionLocal() as db:
        old_token_row = db.scalar(
            select(RefreshToken).where(RefreshToken.token_hash == hash_token(old_refresh))
        )
        assert old_token_row is not None
        assert old_token_row.revoked_at is not None
        old_token_row.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        db.commit()
        user_id = old_token_row.user_id

    client.cookies.set("refresh_token", old_refresh)
    resp = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert resp.status_code == 401

    with SessionLocal() as db:
        all_tokens = db.query(RefreshToken).filter(RefreshToken.user_id == user_id).all()
        assert len(all_tokens) == 2
        assert all(t.revoked_at is not None for t in all_tokens)


def test_reusing_same_old_token_after_rotation_is_caught_not_second_success():
    """Sequential version of the TOCTOU race: present the same still-valid
    token twice. The first call must rotate normally; the second call
    presenting that exact same (now-already-claimed) token must be treated
    as reuse - not silently succeed and mint a second live token."""
    old_refresh = _register("toctou-seq@example.com")

    first = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert first.status_code == 200

    with SessionLocal() as db:
        user_id = db.scalar(
            select(RefreshToken.user_id).where(RefreshToken.token_hash == hash_token(old_refresh))
        )

    client.cookies.set("refresh_token", old_refresh)
    second = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert second.status_code == 401

    with SessionLocal() as db:
        all_tokens = db.query(RefreshToken).filter(RefreshToken.user_id == user_id).all()
        assert len(all_tokens) == 2
        assert all(t.revoked_at is not None for t in all_tokens)


class _BarrierGatedSession:
    """Wraps a real Session but pauses on the FIRST `.scalar()` call made
    through it (the token lookup at the top of `refresh()`) until a partner
    thread's session also completes its first `.scalar()` call, then
    releases both together. Widens the TOCTOU window between "read the
    token's current state" and "act on it" to a size reliably hit by real OS
    threads."""

    def __init__(self, real_session, barrier):
        self._real = real_session
        self._barrier = barrier
        self._gated = False

    def scalar(self, *args, **kwargs):
        result = self._real.scalar(*args, **kwargs)
        if not self._gated:
            self._gated = True
            self._barrier.wait(timeout=5)
            time.sleep(0.05)
        return result

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_concurrent_refresh_with_same_token_only_one_wins_and_chain_is_revoked():
    """True-concurrency version of the TOCTOU race: fire two /auth/refresh
    calls presenting the exact same still-valid token from separate threads."""
    old_refresh = _register("concurrent@example.com")

    with SessionLocal() as db:
        user_id = db.scalar(
            select(RefreshToken.user_id).where(RefreshToken.token_hash == hash_token(old_refresh))
        )

    barrier = threading.Barrier(2)

    def fake_get_session():
        real = SessionLocal()
        try:
            yield _BarrierGatedSession(real, barrier)
        finally:
            real.close()

    app.dependency_overrides[get_session] = fake_get_session
    try:
        results = []

        def do_refresh():
            local_client = TestClient(app, base_url="https://testserver")
            local_client.cookies.set("refresh_token", old_refresh)
            results.append(local_client.post("/auth/refresh", headers=CSRF_HEADERS))

        t1 = threading.Thread(target=do_refresh)
        t2 = threading.Thread(target=do_refresh)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
    finally:
        app.dependency_overrides.pop(get_session, None)

    statuses = sorted(r.status_code for r in results)
    assert statuses == [200, 401]

    with SessionLocal() as db:
        all_tokens = db.query(RefreshToken).filter(RefreshToken.user_id == user_id).all()
        assert all(t.revoked_at is not None for t in all_tokens)
