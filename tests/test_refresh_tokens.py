import threading
import time
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlalchemy import select
from app.main import app
from app.db import SessionLocal, get_session
from app.models.refresh_token import RefreshToken
from app.auth.security import hash_token

client = TestClient(app)


def _register():
    resp = client.post("/auth/register", json={"email": "r@example.com", "password": "pw123456"})
    return resp.json()


def test_refresh_rotates_token():
    tokens = _register()
    resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 200
    new_tokens = resp.json()
    assert new_tokens["refresh_token"] != tokens["refresh_token"]


def test_reused_refresh_token_revokes_chain():
    tokens = _register()
    client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    # reuse the original (now-revoked) token
    resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 401

    with SessionLocal() as db:
        all_tokens = db.query(RefreshToken).all()
        assert all(t.revoked_at is not None for t in all_tokens)


def test_logout_revokes_token():
    tokens = _register()
    client.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 401


def test_reuse_of_already_expired_revoked_token_still_revokes_chain():
    """Regression test (round 2): a delayed-replay attack - steal a token,
    wait out its own TTL, then replay it - must still be caught as reuse
    and revoke the whole chain, not short-circuit into a bare "expired" 401
    that skips chain revocation. This was the precedence bug: checking
    expires_at before attempting the atomic claim let an
    already-revoked-and-now-expired token bypass reuse detection entirely,
    leaving its live sibling token (from the same chain) undetected."""
    tokens = _register()
    old_refresh = tokens["refresh_token"]

    # Rotate normally: old_refresh's row becomes revoked, and a new live
    # token is issued for the same chain.
    first = client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert first.status_code == 200

    # Backdate the now-revoked original token's expires_at into the past,
    # simulating an attacker replaying it long after it would have expired
    # naturally anyway.
    with SessionLocal() as db:
        old_token_row = db.scalar(
            select(RefreshToken).where(RefreshToken.token_hash == hash_token(old_refresh))
        )
        assert old_token_row.revoked_at is not None
        old_token_row.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        db.commit()

    # Replay the old, already-revoked-and-now-expired token.
    resp = client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert resp.status_code == 401

    with SessionLocal() as db:
        all_tokens = db.query(RefreshToken).all()
        # The live token minted by the earlier legitimate rotation must also
        # have been revoked by chain revocation - not just the replayed one.
        assert len(all_tokens) == 2
        assert all(t.revoked_at is not None for t in all_tokens)


def test_reusing_same_old_token_after_rotation_is_caught_not_second_success():
    """Sequential version of the TOCTOU race: present the same still-valid
    token twice. The first call must rotate normally; the second call
    presenting that exact same (now-already-claimed) token must be treated
    as reuse - not silently succeed and mint a second live token. This
    proves the atomic claim-before-mint ordering: if minting happened before
    the old token was atomically claimed, both calls could succeed."""
    tokens = _register()
    old_refresh = tokens["refresh_token"]

    first = client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert first.status_code == 200

    second = client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert second.status_code == 401

    with SessionLocal() as db:
        all_tokens = db.query(RefreshToken).all()
        # Exactly one successful rotation should have happened (the original
        # plus the one token it was rotated into), and the reuse must have
        # revoked the entire chain - including the token minted by the
        # successful first call - since no session can be trusted afterward.
        assert len(all_tokens) == 2
        assert all(t.revoked_at is not None for t in all_tokens)


class _BarrierGatedSession:
    """Wraps a real Session but pauses on the FIRST `.scalar()` call made
    through it (the token lookup at the top of `refresh()`) until a partner
    thread's session also completes its first `.scalar()` call, then
    releases both together. This widens the classic TOCTOU window between
    "read the token's current state" and "act on it" to a size that's
    reliably hit by real OS threads, regardless of GIL/scheduler timing -
    a bare `threading.Barrier` placed *before* the read (e.g. gating
    `hash_token`, which runs before the DB is touched at all) is not
    enough: both requests still each do their own read-then-write in one
    uninterrupted burst afterward, so the race window can easily be missed
    depending on scheduling (verified empirically: an earlier version of
    this test gating only `hash_token` passed even against the pre-fix
    vulnerable code). Gating *after* the read instead guarantees both
    threads have already captured "not revoked yet" before either is
    allowed to act on it - which is exactly the assumption the pre-fix code
    relied on unsafely, and exactly what the atomic claim in the fix
    re-verifies against the live DB row rather than a stale read."""

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
    calls presenting the exact same still-valid token from separate threads,
    with both sessions' first token-lookup forced to interleave via
    `_BarrierGatedSession` (see its docstring for why the barrier has to sit
    *after* the read, not before).

    Before the atomic-claim fix: both requests capture "not revoked" from
    their own read before either has committed a revocation, so both mint a
    new, live refresh token - two live sessions from one presented token,
    silently defeating reuse detection. With the fix: the DB's row-level
    lock on the atomic claim UPDATE re-checks `revoked_at IS NULL` against
    the live row (not the stale read), so only one request can ever claim
    the old token; the loser is treated as reuse, which revokes the whole
    chain (including whatever the winner just minted), so no live session
    survives the race."""
    tokens = _register()
    old_refresh = tokens["refresh_token"]

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
            results.append(client.post("/auth/refresh", json={"refresh_token": old_refresh}))

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
        all_tokens = db.query(RefreshToken).all()
        assert all(t.revoked_at is not None for t in all_tokens)
