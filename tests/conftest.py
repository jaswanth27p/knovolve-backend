import pytest
from sqlalchemy import text
from app.db import SessionLocal


@pytest.fixture(autouse=True)
def clean_db():
    yield
    with SessionLocal() as s:
        # NOTE: brief's snippet also truncates `refresh_tokens`, but that table
        # doesn't exist until Task 3 - only truncating what exists today (`users`).
        # Task 3 should extend this list once `refresh_tokens` is created.
        s.execute(text("TRUNCATE users RESTART IDENTITY CASCADE"))
        s.commit()
