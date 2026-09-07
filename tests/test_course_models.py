from datetime import datetime, timezone
import pytest
from sqlalchemy.exc import IntegrityError
from app.db import SessionLocal
from app.models.course import Course, CourseJob


def _now():
    return datetime.now(timezone.utc)


def test_duplicate_topic_slug_on_courses_rejected():
    with SessionLocal() as db:
        db.add(Course(topic_slug="typescript", topic_raw="TypeScript",
                       topic_embedding=[0.0] * 1024, created_at=_now()))
        db.commit()
        db.add(Course(topic_slug="typescript", topic_raw="TypeScript again",
                       topic_embedding=[0.0] * 1024, created_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_two_active_jobs_same_slug_rejected():
    with SessionLocal() as db:
        db.add(CourseJob(topic_slug="python", topic_raw="Python", topic_embedding=[0.0] * 1024,
                          status="pending", created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(CourseJob(topic_slug="python", topic_raw="Python", topic_embedding=[0.0] * 1024,
                          status="running", created_at=_now(), updated_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_failed_job_does_not_block_new_active_job_same_slug():
    with SessionLocal() as db:
        db.add(CourseJob(topic_slug="rust", topic_raw="Rust", topic_embedding=[0.0] * 1024,
                          status="failed", created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(CourseJob(topic_slug="rust", topic_raw="Rust", topic_embedding=[0.0] * 1024,
                          status="pending", created_at=_now(), updated_at=_now()))
        db.commit()  # should not raise — partial index only covers pending/running
