import hashlib
import secrets
import jwt
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from app.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


def create_access_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_ttl_minutes),
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> int:
    payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return int(payload["sub"])


def create_refresh_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
