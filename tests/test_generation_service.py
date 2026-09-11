from datetime import datetime, timezone
from sqlalchemy import event
from app.db import SessionLocal, engine
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent
from app.models.course import Course, Module, Chapter
from app.models.export import CourseGenerationRun
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


def _seed_global_chapter_with_remediation(user_id=83, slug="generation-global-remediation"):
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        db.add(User(id=user_id, email=f"generation-remediation-{user_id}@example.com", password_hash="x"))
        course = Course(topic_slug=slug, topic_raw="Remediation",
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1, scope="global")
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1, scope="global")
        db.add(chapter)
        db.commit()
        assignment = Assignment(level="module", module_id=module.id, scope="global", status="ready",
                                created_at=now, updated_at=now)
        db.add(assignment)
        db.commit()
        attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user_id, status="graded",
                                    overall_score=0.0, created_at=now, updated_at=now)
        db.add(attempt)
        db.commit()
        base = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                              outline=[], created_at=now, updated_at=now)
        remediation = ChapterContent(chapter_id=chapter.id, version=2, scope="user", user_id=user_id,
                                     status="generating", outline=[],
                                     remediation_source_attempt_id=attempt.id,
                                     created_at=now, updated_at=now)
        db.add_all([base, remediation])
        db.commit()
        db.refresh(course)
        return course.id, remediation.id, chapter.id


def test_plan_includes_user_remediation_for_global_chapter():
    course_id, remediation_id, chapter_id = _seed_global_chapter_with_remediation()
    with SessionLocal() as db:
        from app.models.course import Course as CourseModel
        course = db.get(CourseModel, course_id)
        assert course is not None
        units = svc.plan_units(db, course, 83)
        unit_ids = [u["unit_id"] for u in units]
        assert f"remediation_content:{remediation_id}" in unit_ids
        remediation_unit = next(u for u in units if u["unit_id"] == f"remediation_content:{remediation_id}")
        assert remediation_unit["kind"] == "remediation_content"
        assert remediation_unit["chapter_id"] == chapter_id


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


def test_empty_plan_reuses_completed_check_run_idempotently():
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        db.add(User(id=84, email="generation-complete-idempotent@example.com", password_hash="x"))
        course = Course(topic_slug="generation-complete-idempotent", topic_raw="Complete",
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        db.refresh(course)
        first, first_action = svc.queue_generation_run(db, 84, course)
        second, second_action = svc.queue_generation_run(db, 84, course)
        assert first_action == "already_complete"
        assert second_action == "already_complete"
        assert first.id == second.id
        assert db.query(CourseGenerationRun).filter_by(
            course_id=course.id, user_id=84
        ).count() == 1


def test_get_latest_run_prefers_active_higher_id_on_created_at_tie():
    course_id = _seed_planning_course("generation-latest-tie")
    with SessionLocal() as db:
        from app.models.course import Course as CourseModel
        course = db.get(CourseModel, course_id)
        assert course is not None
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        older = CourseGenerationRun(
            course_id=course.id, user_id=81, status="succeeded", total_units=0, completed_units=0,
            unit_states=[], created_at=created, updated_at=created, completed_at=created,
        )
        db.add(older)
        db.commit()
        active = CourseGenerationRun(
            course_id=course.id, user_id=81, status="running", total_units=1, completed_units=0,
            unit_states=[], created_at=created, updated_at=created,
        )
        db.add(active)
        db.commit()
        latest = svc.get_latest_run(db, 81, course)
        assert latest is not None
        assert latest.id == active.id
        assert latest.id > older.id


def test_queue_generation_run_takes_advisory_lock_before_insert():
    course_id = _seed_planning_course("generation-advisory-lock")
    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", _record)
    try:
        with SessionLocal() as db:
            from app.models.course import Course as CourseModel
            course = db.get(CourseModel, course_id)
            assert course is not None
            svc.queue_generation_run(db, 81, course)
    finally:
        event.remove(engine, "before_cursor_execute", _record)

    lock_indexes = [i for i, s in enumerate(statements) if "pg_advisory_xact_lock" in s]
    assert lock_indexes, f"advisory lock was not issued: {statements}"
    insert_indexes = [
        i for i, s in enumerate(statements) if "INSERT INTO course_generation_runs" in s
    ]
    assert insert_indexes, f"run insert was not issued: {statements}"
    assert lock_indexes[0] < insert_indexes[0]
