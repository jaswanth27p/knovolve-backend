from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal, get_session
from app.models.user import User
from app.auth.security import hash_password

# https base_url so httpx's cookie jar honors the Secure attribute on the
# access/refresh cookies the backend sets — a plain http://testserver would
# silently drop them and every cookie-roundtrip assertion below would fail
# for the wrong reason.
client = TestClient(app, base_url="https://testserver")
CSRF_HEADERS = {"X-Requested-With": "knovolve"}


def test_register_then_login_sets_httponly_cookies_not_body():
    resp = client.post("/auth/register", json={"email": "a@example.com", "password": "hunter22"})
    assert resp.status_code == 201
    assert "access_token" not in resp.json()
    assert "refresh_token" not in resp.json()
    assert client.cookies.get("access_token") is not None
    assert client.cookies.get("refresh_token") is not None
    set_cookie_headers = resp.headers.get_list("set-cookie")
    assert any("access_token" in h and "HttpOnly" in h for h in set_cookie_headers)
    assert any("refresh_token" in h and "HttpOnly" in h for h in set_cookie_headers)

    client.cookies.clear()
    resp = client.post("/auth/login", json={"email": "a@example.com", "password": "hunter22"})
    assert resp.status_code == 200
    assert "access_token" not in resp.json()
    assert client.cookies.get("access_token") is not None
    assert client.cookies.get("refresh_token") is not None


def test_protected_route_works_via_cookie_alone_with_csrf_header():
    client.cookies.clear()
    client.post("/auth/register", json={"email": "cookie-auth@example.com", "password": "hunter22"})
    resp = client.get("/me/courses", headers=CSRF_HEADERS)
    assert resp.status_code == 200


def test_protected_route_via_cookie_without_csrf_header_rejected():
    client.cookies.clear()
    client.post("/auth/register", json={"email": "no-csrf@example.com", "password": "hunter22"})
    resp = client.get("/me/courses")
    assert resp.status_code == 403


def test_login_wrong_password_rejected():
    client.cookies.clear()
    client.post("/auth/register", json={"email": "b@example.com", "password": "correct-horse"})
    resp = client.post("/auth/login", json={"email": "b@example.com", "password": "wrong"})
    assert resp.status_code == 401


def test_duplicate_register_rejected():
    client.cookies.clear()
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
