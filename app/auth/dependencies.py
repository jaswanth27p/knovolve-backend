import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.db import get_session
from app.models.user import User
from app.auth.security import decode_access_token

# auto_error=False: a browser client authenticates via the access_token
# cookie instead of this header, so a missing Authorization header must fall
# through to the cookie check below rather than 401 immediately.
_bearer = HTTPBearer(auto_error=False)

CSRF_HEADER = "x-requested-with"
CSRF_HEADER_VALUE = "knovolve"


def require_csrf_header(request: Request) -> None:
    """SameSite=Strict already blocks the cookie from riding along on a
    cross-site request, but this is defense in depth: a cross-site form post
    (the classic CSRF vector) cannot attach a custom header, and a cross-site
    fetch/XHR that tries to would trigger a CORS preflight our origin
    allowlist rejects. Only requests that are actually same-site can ever
    satisfy this."""
    if request.headers.get(CSRF_HEADER, "").lower() != CSRF_HEADER_VALUE:
        raise HTTPException(status_code=403, detail="missing csrf header")


def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_session),
) -> User:
    if creds:
        # Non-browser client (script, mobile app) authenticating with a
        # bearer header directly - no cookie involved, so no CSRF exposure.
        token = creds.credentials
    else:
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="not authenticated")
        require_csrf_header(request)

    try:
        user_id = decode_access_token(token)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="user not found")
    return user
