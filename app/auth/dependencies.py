import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.db import get_session
from app.models.user import User
from app.auth.security import decode_access_token

_bearer = HTTPBearer()


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_session),
) -> User:
    try:
        user_id = decode_access_token(creds.credentials)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="user not found")
    return user
