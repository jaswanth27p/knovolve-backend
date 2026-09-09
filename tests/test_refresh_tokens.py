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


def test_rotation_inherits_session_started_at():
    """All tokens in one rotation chain share the chain's session birth
    timestamp — it must never drift forward on rotation, or an actively-used
    session could outlive the force-relogin cap forever by constantly
    re-arming its own session age."""
    _register("chain-birth@example.com")
    with SessionLocal() as db:
        first = db.scalar(select(RefreshToken))
        assert first is not None
        birth = first.session_started_at

    for _ in range(3):
        resp = client.post("/auth/refresh", headers=CSRF_HEADERS)
        assert resp.status_code == 200

    with SessionLocal() as db:
        tokens = db.query(RefreshToken).all()
        assert len(tokens) == 4
        assert all(t.session_started_at == birth for t in tokens)


def test_refresh_rejected_when_chain_older_than_force_relogin_cap():
    """The absolute session-age cap: a chain whose session_started_at is past
    force_relogin_after_days must refuse to rotate — even though its sliding
    expires_at is still valid (i.e. the user is actively using the app). The
    whole chain is revoked and the auth cookies cleared so the browser is
    forced to re-login."""
    from app.config import settings

    assert settings.force_relogin_after_days > 0
    _register("force-relogin@example.com")
    with SessionLocal() as db:
        first = db.scalar(select(RefreshToken))
        assert first is not None
        user_id = first.user_id
        # Keep the sliding TTL comfortably valid; only the absolute cap has
        # been exceeded. This distinguishes the new cap from plain expiry.
        first.expires_at = datetime.now(timezone.utc) + timedelta(days=10)
        first.session_started_at = datetime.now(timezone.utc) - timedelta(
            days=settings.force_relogin_after_days + 1
        )
        db.commit()

    resp = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert resp.status_code == 401
    assert client.cookies.get("refresh_token") is None
    assert client.cookies.get("access_token") is None

    with SessionLocal() as db:
        tokens = db.query(RefreshToken).filter(RefreshToken.user_id == user_id).all()
        assert len(tokens) == 1
        assert tokens[0].revoked_at is not None


def test_new_login_starts_a_fresh_session_clock():
    """Logging in again after the cap resets session_started_at to now, so the
    new session gets a full force_relogin_after_days window."""
    from app.config import settings

    register_email = "new-session@example.com"
    _register(register_email)
    with SessionLocal() as db:
        first = db.scalar(select(RefreshToken))
        assert first is not None
        first.session_started_at = datetime.now(timezone.utc) - timedelta(
            days=settings.force_relogin_after_days + 1
        )
        db.commit()

    # Re-login: must issue a fresh session birth, not inherit the old.
    client.cookies.clear()
    resp = client.post("/auth/login", json={"email": register_email, "password": "pw123456"}, headers=CSRF_HEADERS)
    assert resp.status_code == 200
    with SessionLocal() as db:
        tokens = db.query(RefreshToken).all()
        assert len(tokens) == 2
        newest = tokens[-1]
        age = datetime.now(timezone.utc) - newest.session_started_at
        assert age < timedelta(minutes=1)


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


def test_reuse_revokes_only_the_presented_chain_not_other_sessions():
    """Chain-scoped revocation: two independent logins for the SAME user (e.g.
    laptop + phone) form two separate chains. Reuse detected on chain A must
    kill only chain A — chain B (the other device) must keep working. This is
    the regression test for revoking by chain_id instead of by user_id."""
    # Session A: register (chain A)
    client.cookies.clear()
    resp_a = client.post("/auth/register", json={"email": "multi@example.com", "password": "pw123456"}, headers=CSRF_HEADERS)
    assert resp_a.status_code == 201
    token_a = client.cookies.get("refresh_token")
    assert token_a is not None
    with SessionLocal() as db:
        row_a = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(token_a)))
        assert row_a is not None
        chain_a = row_a.chain_id
        user_id = row_a.user_id

    # Session B: re-login same user (chain B)
    client.cookies.clear()
    resp_b = client.post("/auth/login", json={"email": "multi@example.com", "password": "pw123456"}, headers=CSRF_HEADERS)
    assert resp_b.status_code == 200
    token_b = client.cookies.get("refresh_token")
    assert token_b is not None
    with SessionLocal() as db:
        row_b = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(token_b)))
        assert row_b is not None
        chain_b = row_b.chain_id
    assert chain_a != chain_b

    # Simulate reuse on chain A: rotate it (chain A token becomes revoked), then
    # replay the stale chain-A token.
    client.cookies.set("refresh_token", token_a)
    first = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert first.status_code == 200
    client.cookies.set("refresh_token", token_a)
    resp = client.post("/auth/refresh", headers=CSRF_HEADERS)
    assert resp.status_code == 401

    with SessionLocal() as db:
        chain_a_tokens = db.query(RefreshToken).filter(
            RefreshToken.chain_id == chain_a, RefreshToken.user_id == user_id
        ).all()
        chain_b_tokens = db.query(RefreshToken).filter(
            RefreshToken.chain_id == chain_b, RefreshToken.user_id == user_id
        ).all()
        assert len(chain_a_tokens) >= 1
        assert all(t.revoked_at is not None for t in chain_a_tokens)
        assert len(chain_b_tokens) >= 1
        assert all(t.revoked_at is None for t in chain_b_tokens)


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
