from datetime import datetime, timedelta, timezone
from typing import cast

from sqlalchemy import CursorResult, delete, or_

from app.config import settings
from app.db import SessionLocal
from app.models.refresh_token import RefreshToken
from app.tasks.celery_app import celery_app


def purge_expired_refresh_tokens() -> int:
    """Delete refresh_tokens rows that can no longer be presented to
    /auth/refresh: either revoked (rotated out or reuse-chain-killed) long
    enough ago that reuse-detection forensics no longer need the row, or
    expired without ever having been rotated (abandoned after issuance).
    Returns the number of rows deleted."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.refresh_token_purge_after_days)
    with SessionLocal() as db:
        # Session.execute() is typed to return the generic Result[Any] base
        # class, but a Core DELETE statement always executes through the
        # DBAPI cursor and returns a CursorResult, which is what actually
        # carries `rowcount`.
        result = cast(CursorResult, db.execute(
            delete(RefreshToken).where(
                or_(
                    RefreshToken.revoked_at < cutoff,
                    RefreshToken.expires_at < cutoff,
                )
            )
        ))
        db.commit()
        return result.rowcount


@celery_app.task
def purge_expired_refresh_tokens_task() -> int:
    return purge_expired_refresh_tokens()
