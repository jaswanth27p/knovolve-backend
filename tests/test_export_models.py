from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.db import SessionLocal
from app.models.course import Course
from app.models.export import CourseGenerationRun, ExportJob
from app.models.user import User


@pytest.fixture(autouse=True)
def _seed_export_user():
    with SessionLocal() as db:
        db.add(User(id=41, email="export-model-user@example.com", password_hash="x"))
        db.commit()


def _make_course(slug):
    with SessionLocal() as db:
        course = Course(
            topic_slug=slug,
            topic_raw="Export Model",
            topic_embedding=[0.0] * 2048,
            created_at=datetime.now(timezone.utc),
        )
        db.add(course)
        db.commit()
        db.refresh(course)
        return course.id


def test_export_job_defaults_to_pending_without_result():
    course_id = _make_course("export-model-course")
    with SessionLocal() as db:
        job = ExportJob(
            course_id=course_id,
            user_id=41,
            kind="course",
            status="pending",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        assert job.id is not None
        assert job.result_key is None
        assert job.completed_at is None


def test_only_one_active_generation_run_per_course_and_user():
    course_id = _make_course("generation-model-course")
    with SessionLocal() as db:
        db.add(CourseGenerationRun(
            course_id=course_id,
            user_id=41,
            status="running",
            total_units=3,
            completed_units=0,
            unit_states=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        db.commit()
        db.add(CourseGenerationRun(
            course_id=course_id,
            user_id=41,
            status="pending",
            total_units=3,
            completed_units=0,
            unit_states=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ))
        with pytest.raises(IntegrityError):
            db.commit()
