from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal
from app.models.refresh_token import RefreshToken

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
