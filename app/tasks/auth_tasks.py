from datetime import datetime, timedelta, timezone
from typing import cast

from sqlalchemy import CursorResult, and_, delete, not_, or_, select

from app.config import settings
from app.db import SessionLocal
from app.models.refresh_token import RefreshToken
from app.tasks.celery_app import celery_app


def purge_expired_refresh_tokens() -> int:
    """Delete refresh_tokens rows that can no longer be presented to
    /auth/refresh: either revoked (rotated out or reuse-chain-killed) long
    enough ago AND belonging to a chain that has no live (unrevoked) token
    left, or expired without ever having been rotated (abandoned after
    issuance). Returns the number of rows deleted.

    The "chain has no live token left" clause is load-bearing, not
    incidental: /auth/refresh's reuse detection (see auth/routes.py) depends
    on a chain's earlier revoked rows still being present to recognize a
    replayed stolen token and revoke that chain's current live sibling. A
    time-only cutoff (just `revoked_at < cutoff`) is only safe for a still-
    live chain if `refresh_token_purge_after_days >= force_relogin_after_days`
    — true only by coincidence of both defaulting to 30 today, and not true
    at all when force_relogin_after_days is disabled (0), where a
    continuously-active chain has no age bound whatsoever. Gating deletion on
    "this chain is entirely dead" instead makes the purge correct regardless
    of how those two settings are tuned relative to each other."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.refresh_token_purge_after_days)
    with SessionLocal() as db:
        live_chain_ids = select(RefreshToken.chain_id).where(RefreshToken.revoked_at.is_(None))
        # Session.execute() is typed to return the generic Result[Any] base
        # class, but a Core DELETE statement always executes through the
        # DBAPI cursor and returns a CursorResult, which is what actually
        # carries `rowcount`.
        result = cast(CursorResult, db.execute(
            delete(RefreshToken).where(
                or_(
                    and_(
                        RefreshToken.revoked_at.is_not(None),
                        RefreshToken.revoked_at < cutoff,
                        not_(RefreshToken.chain_id.in_(live_chain_ids)),
                    ),
                    and_(
                        RefreshToken.revoked_at.is_(None),
                        RefreshToken.expires_at < cutoff,
                    ),
                )
            )
        ))
        db.commit()
        return result.rowcount


@celery_app.task
def purge_expired_refresh_tokens_task() -> int:
    return purge_expired_refresh_tokens()
