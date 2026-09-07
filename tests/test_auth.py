from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal, get_session
from app.models.user import User
from app.auth.security import hash_password

client = TestClient(app)


def test_register_then_login():
    resp = client.post("/auth/register", json={"email": "a@example.com", "password": "hunter22"})
    assert resp.status_code == 201
    assert "access_token" in resp.json()

    resp = client.post("/auth/login", json={"email": "a@example.com", "password": "hunter22"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password_rejected():
    client.post("/auth/register", json={"email": "b@example.com", "password": "correct-horse"})
    resp = client.post("/auth/login", json={"email": "b@example.com", "password": "wrong"})
    assert resp.status_code == 401


def test_duplicate_register_rejected():
    client.post("/auth/register", json={"email": "c@example.com", "password": "x"})
    resp = client.post("/auth/register", json={"email": "c@example.com", "password": "y"})
    assert resp.status_code == 409


def test_login_nonexistent_user_rejected():
    resp = client.post("/auth/login", json={"email": "no-such-user@example.com", "password": "whatever"})
    assert resp.status_code == 401


class _AlwaysMissSession:
    """Wraps a real Session but forces .scalar() to always return None,
    simulating the TOCTOU window in register(): the pre-check SELECT finds
    nothing (as if a concurrent request hadn't committed yet), but the
    underlying DB already has a real row with the same email, so the
    subsequent insert+commit hits the real unique constraint."""

    def __init__(self, real_session):
        self._real = real_session

    def scalar(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_register_race_condition_returns_409_not_500():
    # Seed a user directly at the DB layer - bypassing register()'s
    # pre-check entirely - so the only thing that can catch the duplicate
    # is the unique constraint hit during commit().
    with SessionLocal() as s:
        s.add(User(email="race@example.com", password_hash=hash_password("x")))
        s.commit()

    def fake_get_session():
        real = SessionLocal()
        try:
            yield _AlwaysMissSession(real)
        finally:
            real.close()

    app.dependency_overrides[get_session] = fake_get_session
    try:
        resp = client.post("/auth/register", json={"email": "race@example.com", "password": "y"})
    finally:
        app.dependency_overrides.pop(get_session, None)

    assert resp.status_code == 409
