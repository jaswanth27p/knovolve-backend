import pytest
from sqlalchemy import text
from app.db import SessionLocal


@pytest.fixture(autouse=True)
def clean_db():
    yield
    with SessionLocal() as s:
        s.execute(text("TRUNCATE refresh_tokens, users RESTART IDENTITY CASCADE"))
        s.commit()
