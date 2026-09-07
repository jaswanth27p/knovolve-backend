from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.db import get_session
from app.models.user import User
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.auth.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    """A precomputed bcrypt hash of a fixed placeholder, verified against on
    login when no matching user exists. This makes the "no such user" and
    "wrong password" paths pay the same bcrypt cost, closing a timing side
    channel that would otherwise let an attacker enumerate registered
    emails by measuring response latency."""
    return hash_password("no-such-user-timing-equalization-placeholder")


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_session)):
    existing = db.scalar(select(User).where(User.email == body.email))
    if existing:
        raise HTTPException(status_code=409, detail="email already registered")
    user = User(email=body.email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Belt-and-suspenders for the pre-check above: two concurrent
        # registrations for the same email can both pass the SELECT before
        # either commits, so the unique constraint on users.email is the
        # real source of truth here.
        db.rollback()
        raise HTTPException(status_code=409, detail="email already registered")
    db.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_session)):
    user = db.scalar(select(User).where(User.email == body.email))
    # Always run a bcrypt verify, even when no user is found, so a "no such
    # user" 401 and a "wrong password" 401 take the same amount of time.
    password_hash = user.password_hash if user else _dummy_password_hash()
    password_ok = verify_password(body.password, password_hash)
    if not user or not password_ok:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return TokenResponse(access_token=create_access_token(user.id))
