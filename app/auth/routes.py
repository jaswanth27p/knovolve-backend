from functools import lru_cache
from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.config import settings
from app.db import get_session
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.schemas.auth import RegisterRequest, LoginRequest, AuthResponse
from app.auth.dependencies import require_csrf_header
from app.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    hash_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"


@lru_cache(maxsize=1)
def _dummy_password_hash() -> str:
    """A precomputed bcrypt hash of a fixed placeholder, verified against on
    login when no matching user exists. This makes the "no such user" and
    "wrong password" paths pay the same bcrypt cost, closing a timing side
    channel that would otherwise let an attacker enumerate registered
    emails by measuring response latency."""
    return hash_password("no-such-user-timing-equalization-placeholder")


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    # httpOnly + Secure + SameSite=Strict: JS (including anything an XSS
    # payload runs) can never read either cookie, and the browser will not
    # attach them to any cross-site request. refresh_token is additionally
    # scoped to /auth so it never leaves the browser on ordinary API calls -
    # only the two auth endpoints that need it ever see it.
    response.set_cookie(
        ACCESS_COOKIE, access_token,
        max_age=settings.jwt_access_ttl_minutes * 60,
        httponly=True, secure=True, samesite=settings.cookie_samesite, path="/",
    )
    response.set_cookie(
        REFRESH_COOKIE, refresh_token,
        max_age=settings.jwt_refresh_ttl_days * 24 * 60 * 60,
        httponly=True, secure=True, samesite=settings.cookie_samesite, path="/auth",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path="/auth")


def _issue_tokens(db: Session, user: User, response: Response) -> None:
    now = datetime.now(timezone.utc)
    raw_refresh = create_refresh_token()
    db.add(RefreshToken(
        user_id=user.id,
        token_hash=hash_token(raw_refresh),
        expires_at=now + timedelta(days=settings.jwt_refresh_ttl_days),
        created_at=now,
        # A fresh login/registration is a brand-new session: its birth clock
        # starts now. Rotations later inherit this, so the whole chain shares
        # one session_started_at and force_relogin_after_days is measured from
        # the login, not from each individual refresh.
        session_started_at=now,
        # A new login/registration mints its own unique chain identity,
        # inherited by every rotation in the chain. Reuse detection and the
        # forced-relogin cap revoke by this ID, so only this login is killed —
        # a different login of the same user (another device) survives.
        chain_id=str(uuid4()),
    ))
    db.commit()
    _set_auth_cookies(response, create_access_token(user.id), raw_refresh)


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(body: RegisterRequest, response: Response, db: Session = Depends(get_session)):
    # Accepted tradeoff: a distinct 409 here is a user-enumeration oracle
    # (an attacker can learn which emails are registered). This is standard
    # practice for products where "that email is taken" is expected UX during
    # signup, and it's a separate concern from login()'s timing equalization
    # below, which defends the credential itself rather than account
    # existence. Closing this would mean a generic response on both outcomes,
    # which is a real UX tradeoff, not a bug fix — left as-is deliberately.
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
    _issue_tokens(db, user, response)
    return AuthResponse()


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, response: Response, db: Session = Depends(get_session)):
    user = db.scalar(select(User).where(User.email == body.email))
    # Always run a bcrypt verify, even when no user is found, so a "no such
    # user" 401 and a "wrong password" 401 take the same amount of time.
    password_hash = user.password_hash if user else _dummy_password_hash()
    password_ok = verify_password(body.password, password_hash)
    if not user or not password_ok:
        raise HTTPException(status_code=401, detail="invalid credentials")
    _issue_tokens(db, user, response)
    return AuthResponse()


def _revoke_chain(db: Session, chain_id: str) -> None:
    """Revoke every currently-active refresh token in one login chain (identified
    by chain_id). Called when reuse of an already-claimed/revoked token is
    detected, since at that point we can no longer tell which token in the chain
    is the legitimate one. Scoped to the chain, not the user: a separate login
    of the same user (e.g. another device) has its own chain_id and survives."""
    db.execute(
        update(RefreshToken)
        .where(RefreshToken.chain_id == chain_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(timezone.utc))
    )
    db.commit()


@router.post("/refresh", response_model=AuthResponse, dependencies=[Depends(require_csrf_header)])
def refresh(request: Request, response: Response, db: Session = Depends(get_session)):
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    if not raw_refresh:
        raise HTTPException(status_code=401, detail="no refresh token")
    token_hash = hash_token(raw_refresh)
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
        # Either way, treat it as reuse: revoke the chain - we can no longer
        # tell which branch is legitimate. Because the winning claim+mint+link
        # below is one atomic transaction gated by the same row lock, if a
        # winner exists its new token is guaranteed to already be committed and
        # visible by the time we get here. Only this login chain is revoked; a
        # separate login (another device) survives.
        db.rollback()
        _revoke_chain(db, token.chain_id)
        raise HTTPException(status_code=401, detail="refresh token reuse detected")

    if token.expires_at < datetime.now(timezone.utc):
        # This token was never rotated or reused before now - it was simply
        # never used again after issuance and its TTL ran out. We've already
        # claimed/revoked it above, which is correct (an expired token
        # shouldn't be usable to rotate anyway); just persist that and
        # reject.
        db.commit()
        raise HTTPException(status_code=401, detail="refresh token expired")

    # Absolute session-age cap (force_relogin_after_days): the whole chain
    # shares one birth timestamp, so an actively-used session is still forced
    # to re-login after the cap even though each individual token's sliding
    # expires_at keeps renewing. The presented token is already claimed
    # (revoked) by the UPDATE above; kill the rest of the chain and clear the
    # cookies so the client cannot keep presenting siblings past the cap.
    if (
        settings.force_relogin_after_days > 0
        and datetime.now(timezone.utc) - token.session_started_at
        >= timedelta(days=settings.force_relogin_after_days)
    ):
        _revoke_chain(db, token.chain_id)
        # A raised HTTPException would drop the delete-cookie headers (FastAPI
        # builds the error response fresh, ignoring headers set on the
        # injected `response` object), leaving stale cookies in the browser.
        # Return a JSONResponse with the clear-cookie headers attached to the
        # error response itself so the client actually gets logged out.
        from fastapi.responses import JSONResponse

        error = JSONResponse(status_code=401, content={"detail": "session expired, please log in again"})
        error.delete_cookie(ACCESS_COOKIE, path="/")
        error.delete_cookie(REFRESH_COOKIE, path="/auth")
        return error

    user = db.get(User, token.user_id)
    # token.user_id is a foreign key to users.id and there is no user-deletion
    # path in this codebase, so the referenced user is guaranteed to exist.
    assert user is not None
    new_raw_refresh = create_refresh_token()
    now = datetime.now(timezone.utc)
    new_token_row = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(new_raw_refresh),
        expires_at=now + timedelta(days=settings.jwt_refresh_ttl_days),
        created_at=now,
        # Inherit the chain's session birth so the absolute cap is measured
        # from the original login, not reset by each rotation (which would let
        # an active session sidestep the cap forever).
        session_started_at=token.session_started_at,
        # Stay in the same login chain so chain-scoped revocation (reuse
        # detection) and the forced-relogin cap apply to the whole session.
        chain_id=token.chain_id,
    )
    db.add(new_token_row)
    # Caller is holding a row lock from the claim UPDATE above and needs the
    # new row's id before it commits. flush() assigns the id and makes the
    # insert visible to later statements in this transaction without
    # releasing the lock.
    db.flush()
    token.replaced_by_id = new_token_row.id
    db.commit()
    _set_auth_cookies(response, create_access_token(user.id), new_raw_refresh)
    return AuthResponse()


@router.post("/logout", status_code=204, dependencies=[Depends(require_csrf_header)])
def logout(request: Request, response: Response, db: Session = Depends(get_session)):
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    if raw_refresh:
        token_hash = hash_token(raw_refresh)
        token = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
        if token and token.revoked_at is None:
            token.revoked_at = datetime.now(timezone.utc)
            db.commit()
    _clear_auth_cookies(response)
