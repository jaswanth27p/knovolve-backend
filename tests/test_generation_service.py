from datetime import datetime, timezone
from fastapi import HTTPException
import pytest
from app.db import SessionLocal
from app.models.assignment import Assignment
from app.models.chapter_content import ChapterContent
from app.models.course import Course, Module, Chapter
from app.models.user import User
from app.services import generation as svc


def _seed_planning_course(slug="generation-planning"):
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        db.add(User(id=81, email="generation-planning@example.com", password_hash="x"))
        course = Course(topic_slug=slug, topic_raw="Planning",
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1, scope="global")
        db.add(module)
        db.commit()
        first = Chapter(module_id=module.id, title="First", objective="o", order=1, scope="global")
        second = Chapter(module_id=module.id, title="Second", objective="o", order=2, scope="global")
        db.add_all([first, second])
        db.commit()
        ready = ChapterContent(chapter_id=first.id, version=1, scope="global", status="ready",
                               outline=[], created_at=now, updated_at=now)
        incomplete = ChapterContent(chapter_id=second.id, version=1, scope="global", status="failed",
                                    outline=[], created_at=now, updated_at=now)
        db.add_all([ready, incomplete])
        db.commit()
        db.refresh(course)
        return course.id


def test_plan_includes_ready_assignment_work_and_omits_unready_module_assignment():
    course_id = _seed_planning_course()
    with SessionLocal() as db:
        from app.models.course import Course as CourseModel
        course = db.get(CourseModel, course_id)
        assert course is not None
        ready_content = db.query(ChapterContent).filter_by(status="ready").one()
        units = svc.plan_units(db, course, 81)
        assert [u["unit_id"] for u in units] == [
            f"chapter_assignment:{ready_content.id}",
            "content:2",
        ]
        assert units[0]["kind"] == "chapter_assignment"
        assert units[1]["kind"] == "content"


def test_duplicate_active_run_returns_existing_row():
    course_id = _seed_planning_course("generation-duplicate")
    with SessionLocal() as db:
        from app.models.course import Course as CourseModel
        course = db.get(CourseModel, course_id)
        assert course is not None
        first, first_action = svc.queue_generation_run(db, 81, course)
        second, second_action = svc.queue_generation_run(db, 81, course)
        assert first_action == "queued"
        assert second_action == "already_running"
        assert first.id == second.id


def test_empty_plan_records_completed_check_run():
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        db.add(User(id=82, email="generation-complete@example.com", password_hash="x"))
        course = Course(topic_slug="generation-complete", topic_raw="Complete",
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        db.refresh(course)
        run, action = svc.queue_generation_run(db, 82, course)
        assert action == "already_complete"
        assert run.status == "succeeded"
        assert run.total_units == 0
