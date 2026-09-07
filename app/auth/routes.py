from functools import lru_cache
from datetime import datetime, timedelta, timezone
from typing import cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.config import settings
from app.db import get_session
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, RefreshRequest
from app.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    hash_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    """A precomputed bcrypt hash of a fixed placeholder, verified against on
    login when no matching user exists. This makes the "no such user" and
    "wrong password" paths pay the same bcrypt cost, closing a timing side
    channel that would otherwise let an attacker enumerate registered
    emails by measuring response latency."""
    return hash_password("no-such-user-timing-equalization-placeholder")


def _issue_tokens(db: Session, user: User, commit: bool = True) -> TokenResponse:
    raw_refresh = create_refresh_token()
    db.add(RefreshToken(
        user_id=user.id,
        token_hash=hash_token(raw_refresh),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_ttl_days),
        created_at=datetime.now(timezone.utc),
    ))
    if commit:
        db.commit()
    else:
        # Caller (refresh()) is holding a row lock from an earlier statement
        # in this same transaction and needs the new row's id before it
        # commits. flush() assigns the id and makes the insert visible to
        # later statements in this transaction without releasing the lock.
        db.flush()
    return TokenResponse(access_token=create_access_token(user.id), refresh_token=raw_refresh)


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
    return _issue_tokens(db, user)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_session)):
    user = db.scalar(select(User).where(User.email == body.email))
    # Always run a bcrypt verify, even when no user is found, so a "no such
    # user" 401 and a "wrong password" 401 take the same amount of time.
    password_hash = user.password_hash if user else _dummy_password_hash()
    password_ok = verify_password(body.password, password_hash)
    if not user or not password_ok:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return _issue_tokens(db, user)


def _revoke_active_chain(db: Session, user_id: int) -> None:
    """Revoke every currently-active refresh token for this user. Called when
    reuse of an already-claimed/revoked token is detected, since at that point
    we can no longer tell which token in the chain is the legitimate one."""
    db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(timezone.utc))
    )
    db.commit()


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_session)):
    token_hash = hash_token(body.refresh_token)
    token = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    if not token:
        raise HTTPException(status_code=401, detail="invalid refresh token")

    # Atomically claim this token BEFORE checking expiry, and unconditionally
    # of it: the UPDATE's WHERE clause (id + revoked_at IS NULL) takes a row
    # lock and only succeeds for whichever concurrent request gets there
    # first. Precedence matters here, not just timing - if we checked
    # expires_at first, a token that was already revoked by a legitimate
    # prior rotation (or a genuine reuse attempt) AND has since crossed its
    # own TTL would short-circuit into a bare "expired" 401 and never reach
    # chain revocation below, silently leaving that chain's live sibling
    # token undetected (the classic "steal a token, wait out its TTL, then
    # replay it" attack). Attempting the claim first means an
    # already-revoked token - expired or not - always routes to reuse
    # detection.
    #
    # We deliberately do NOT commit the claim on its own yet - the row lock
    # (and thus mutual exclusion against a concurrent request presenting the
    # same token) must be held for the whole claim+mint+link unit of work
    # below, committed exactly once at the end. If we committed the claim
    # alone, releasing the lock early, a losing request could run its reuse
    # chain-revoke query in the gap before this transaction's new token is
    # committed, and miss revoking it - reopening a narrower version of the
    # exact bug this fix targets.
    # Session.execute() is typed to return the generic Result[Any] base class
    # (it also covers ORM-returning statements), but a Core UPDATE statement
    # always executes through the DBAPI cursor and returns a CursorResult,
    # which is what actually carries `rowcount`.
    claim = cast(CursorResult, db.execute(
        update(RefreshToken)
        .where(RefreshToken.id == token.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(timezone.utc))
    ))
    if claim.rowcount == 0:
        # Someone else already claimed/rotated this token first (lost a
        # concurrent race - by the time we got the row lock, theirs had
        # already committed), or this is a genuine reuse of an
        # already-revoked token (whether or not it has also expired since).
        # Either way, treat it as reuse: revoke the user's whole active
        # chain - we can no longer tell which branch is legitimate. Because
        # the winning claim+mint+link below is one atomic transaction gated
        # by the same row lock, if a winner exists its new token is
        # guaranteed to already be committed and visible by the time we get
        # here.
        db.rollback()
        _revoke_active_chain(db, token.user_id)
        raise HTTPException(status_code=401, detail="refresh token reuse detected")

    if token.expires_at < datetime.now(timezone.utc):
        # This token was never rotated or reused before now - it was simply
        # never used again after issuance and its TTL ran out. We've already
        # claimed/revoked it above, which is correct (an expired token
        # shouldn't be usable to rotate anyway); just persist that and
        # reject.
        db.commit()
        raise HTTPException(status_code=401, detail="refresh token expired")

    user = db.get(User, token.user_id)
    # token.user_id is a foreign key to users.id and there is no user-deletion
    # path in this codebase, so the referenced user is guaranteed to exist.
    assert user is not None
    new_tokens = _issue_tokens(db, user, commit=False)
    new_token_row = db.scalar(select(RefreshToken).where(
        RefreshToken.token_hash == hash_token(new_tokens.refresh_token)
    ))
    # _issue_tokens(commit=False) just inserted and flushed this exact row in
    # this same transaction, so it is guaranteed to be found here.
    assert new_token_row is not None
    token.replaced_by_id = new_token_row.id
    db.commit()
    return new_tokens


@router.post("/logout", status_code=204)
def logout(body: RefreshRequest, db: Session = Depends(get_session)):
    token_hash = hash_token(body.refresh_token)
    token = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    if token and token.revoked_at is None:
        token.revoked_at = datetime.now(timezone.utc)
        db.commit()
