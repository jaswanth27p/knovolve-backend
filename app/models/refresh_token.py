from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    replaced_by_id: Mapped[int | None] = mapped_column(ForeignKey("refresh_tokens.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Birth of the login session this token belongs to. Every token in a
    # rotation chain shares the same value (set at login/register, inherited
    # on rotation), so /auth/refresh can enforce an absolute session-age cap
    # (force_relogin_after_days) with a single column read instead of walking
    # the replaced_by_id chain.
    session_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Identity of the login "session" this token belongs to, shared by every
    # rotation in its chain. Unlike session_started_at (a timestamp, which
    # two rapid logins could theoretically collide on), chain_id is a
    # guaranteed-unique token minted at login and inherited on rotation. Chain
    # revocation/reuse-detection targets this ID so only the compromised login
    # is killed - a re-login on another device shares the user_id but NOT the
    # chain_id, and must survive an unrelated reuse event.
    chain_id: Mapped[str] = mapped_column(String(36), index=True)
