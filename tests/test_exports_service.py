from datetime import datetime, timezone
from typing import cast
from unittest.mock import patch
from fastapi import HTTPException
import pytest
from sqlalchemy import select
from app.db import SessionLocal
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Course, Module, Chapter
from app.models.user import User
from app.services import exports as svc


def test_scope_helpers_separate_global_and_user_versions():
    course_id = _seed_gate_course("export-scope-structure")
    with SessionLocal() as db:
        course = db.get(Course, course_id)
        assert course is not None
        first = db.query(Chapter).filter_by(title="First").one()
        extra = db.query(Chapter).filter_by(title="Extra").one()
        assert [c.version for c in svc.visible_versions(db, first, 51, include_remediation=False)] == [1]
        assert [c.version for c in svc.visible_versions(db, first, 51, include_remediation=True)] == [1, 2]
        assert [c.version for c in svc.visible_versions(db, first, 999, include_remediation=True)] == [1]
        assert [c.version for c in svc.visible_versions(db, extra, 51, include_remediation=True)] == [1]


def test_version_label_omits_personalized_prefix_for_remediation():
    course_id = _seed_gate_course("export-version-label")
    with SessionLocal() as db:
        course = db.get(Course, course_id)
        assert course is not None
        first = db.query(Chapter).filter_by(title="First").one()
        base = svc.chapter_base_content(db, first, 51)
        assert base is not None
        remediation = db.scalars(
            select(ChapterContent).where(
                ChapterContent.chapter_id == first.id,
                ChapterContent.remediation_source_attempt_id.isnot(None),
            )
        ).one()
        assert svc._version_label(base) == f"Version {base.version}"
        assert svc._version_label(remediation) == f"Version {remediation.version}"


def _detail_code(exc: pytest.ExceptionInfo[HTTPException]) -> str:
    return cast(dict, exc.value.detail)["code"]


def _add_section(db, content, order=0):
    db.add(ChapterContentSection(
        chapter_content_id=content.id,
        order=order,
        heading="Section",
        kind="teaching",
        body_markdown="Body",
        examples=[{"prompt": "P", "walkthrough": "W"}],
    ))


def _add_content(db, chapter, version, scope, user_id, remediation_source_attempt_id, status="ready"):
    content = ChapterContent(
        chapter_id=chapter.id,
        version=version,
        scope=scope,
        user_id=user_id,
        status=status,
        outline=[],
        remediation_target_tags=[] if scope == "user" else None,
        remediation_source_attempt_id=remediation_source_attempt_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(content)
    db.commit()
    db.refresh(content)
    _add_section(db, content)
    db.commit()
    return content


def _add_assignment(db, level, status, scope, user_id, content_id=None, module_id=None):
    assignment = Assignment(
        level=level,
        chapter_content_id=content_id,
        module_id=module_id,
        scope=scope,
        user_id=user_id,
        status=status,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    db.add(AssignmentQuestion(
        assignment_id=assignment.id,
        order=1,
        type="mcq",
        text="Question",
        options=["A", "B"],
        correct_answer="A",
        explanation="Because A is correct.",
        concept_tag="concept",
        difficulty="easy",
    ))
    db.commit()
    return assignment


def _seed_gate_course(slug="export-gates"):
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        db.add(User(id=51, email="export-gates-51@example.com", password_hash="x"))
        course = Course(topic_slug=slug, topic_raw="Gates",
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module_one = Module(course_id=course.id, title="Global One", objective="o",
                            order=1, scope="global")
        module_two = Module(course_id=course.id, title="Global Two", objective="o",
                            order=2, scope="global")
        db.add_all([module_one, module_two])
        db.commit()
        first = Chapter(module_id=module_one.id, title="First", objective="o", order=1, scope="global")
        second = Chapter(module_id=module_one.id, title="Second", objective="o", order=2, scope="global")
        third = Chapter(module_id=module_two.id, title="Third", objective="o", order=1, scope="global")
        db.add_all([first, second, third])
        db.commit()
        bucket = Module(course_id=course.id, title="Additional Chapters", objective="o",
                        order=3, scope="user", user_id=51)
        db.add(bucket)
        db.commit()
        extra = Chapter(module_id=bucket.id, title="Extra", objective="o", order=1,
                        scope="user", user_id=51)
        db.add(extra)
        db.commit()

        first_content = _add_content(db, first, 1, "global", None, None)
        second_content = _add_content(db, second, 1, "global", None, None)
        third_content = _add_content(db, third, 1, "global", None, None)
        extra_content = _add_content(db, extra, 1, "user", 51, None)
        first_assignment = _add_assignment(db, "chapter", "ready", "global", None, content_id=first_content.id)
        _add_assignment(db, "chapter", "ready", "global", None, content_id=second_content.id)
        _add_assignment(db, "chapter", "ready", "global", None, content_id=third_content.id)
        _add_assignment(db, "chapter", "ready", "user", 51, content_id=extra_content.id)
        _add_assignment(db, "module", "ready", "global", None, module_id=module_one.id)
        _add_assignment(db, "module", "ready", "global", None, module_id=module_two.id)

        attempt = AssignmentAttempt(
            assignment_id=first_assignment.id,
            user_id=51,
            status="graded",
            overall_score=0.4,
            created_at=now,
            updated_at=now,
        )
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
        remediated = _add_content(db, first, 2, "user", 51, attempt.id)
        _add_assignment(db, "chapter", "ready", "user", 51, content_id=remediated.id)
        db.refresh(course)
        return course.id


def test_all_export_gates_pass_for_fully_ready_course():
    course_id = _seed_gate_course()
    with SessionLocal() as db:
        course = db.get(Course, course_id)
        assert course is not None
        for kind in ("course", "assignments", "full_course", "full_assignments", "custom"):
            svc.require_export_ready(db, course, 51, kind)


def test_incomplete_triggered_remediation_blocks_full_and_custom_exports():
    course_id = _seed_gate_course("export-remediation-gate")
    with SessionLocal() as db:
        course = db.get(Course, course_id)
        assert course is not None
        remediated = db.scalars(
            select(ChapterContent).where(
                ChapterContent.remediation_source_attempt_id.isnot(None),
                ChapterContent.user_id == 51,
            )
        ).one()
        remediated.status = "failed"
        db.commit()

        with pytest.raises(HTTPException) as exc:
            svc.require_export_ready(db, course, 51, "full_course")
        assert exc.value.status_code == 409
        assert _detail_code(exc) == "versions_not_ready"

        with pytest.raises(HTTPException) as exc:
            svc.require_export_ready(db, course, 51, "custom")
        assert _detail_code(exc) == "versions_not_ready"

        # Global-only exports do not depend on remediation completeness.
        svc.require_export_ready(db, course, 51, "course")
        svc.require_export_ready(db, course, 51, "assignments")


def test_failed_module_assignment_blocks_assignment_exports():
    course_id = _seed_gate_course("export-assignment-gate")
    with SessionLocal() as db:
        course = db.get(Course, course_id)
        assert course is not None
        module_assignment = db.scalar(
            select(Assignment).where(
                Assignment.level == "module",
                Assignment.scope == "global",
            )
        )
        assert module_assignment is not None
        module_assignment.status = "failed"
        db.commit()

        with pytest.raises(HTTPException) as exc:
            svc.require_export_ready(db, course, 51, "assignments")
        assert exc.value.status_code == 409
        assert _detail_code(exc) == "assignments_not_ready"

        with pytest.raises(HTTPException) as exc:
            svc.require_export_ready(db, course, 51, "full_assignments")
        assert _detail_code(exc) == "assignments_not_ready"

        # Course-content exports do not depend on assignment readiness.
        svc.require_export_ready(db, course, 51, "course")
        svc.require_export_ready(db, course, 51, "full_course")


def test_missing_global_content_blocks_every_kind():
    course_id = _seed_gate_course("export-content-gate")
    with SessionLocal() as db:
        course = db.get(Course, course_id)
        assert course is not None
        global_content = db.scalar(
            select(ChapterContent).where(
                ChapterContent.scope == "global",
                ChapterContent.status == "ready",
            )
        )
        assert global_content is not None
        global_content.status = "failed"
        db.commit()

        for kind in ("course", "assignments", "full_course", "full_assignments", "custom"):
            with pytest.raises(HTTPException) as exc:
                svc.require_export_ready(db, course, 51, kind)
            assert exc.value.status_code == 409
            assert _detail_code(exc) == "content_not_ready"


def test_custom_export_requires_brief_and_plan():
    with SessionLocal() as db:
        course = db.query(Course).first()
        with pytest.raises(HTTPException) as exc:
            svc.create_export_job(db, 51, course, "custom", None)  # pyright: ignore[reportArgumentType]
        assert exc.value.status_code == 400


def test_retry_export_job_reuses_params_and_rejects_non_failed():
    course_id = _seed_gate_course("export-retry-service")
    with SessionLocal() as db:
        course = db.get(Course, course_id)
        assert course is not None
        plan = {
            "title": "Custom doc",
            "output_kind": "summary",
            "length": "short",
            "item_count": None,
            "notes": None,
        }
        failed = svc.create_export_job(db, 51, course, "custom", {"brief": "summarize", "plan": plan})
        failed.status = "failed"
        db.commit()

        clone = svc.retry_export_job(db, 51, course, failed.id)
        assert clone.id != failed.id
        assert clone.kind == "custom"
        assert clone.params is not None
        assert clone.params["brief"] == "summarize"
        assert clone.params["plan"]["title"] == "Custom doc"

        clone.status = "succeeded"
        db.commit()
        with pytest.raises(HTTPException) as exc:
            svc.retry_export_job(db, 51, course, clone.id)
        assert exc.value.status_code == 409
        assert _detail_code(exc) == "not_retryable"


def test_run_clarify_wraps_agent_parse_failure_as_retryable_502():
    course_id = _seed_gate_course("export-clarify-failure")
    with SessionLocal() as db:
        course = db.get(Course, course_id)
        assert course is not None
        with patch.object(svc, "clarify_export", side_effect=ValueError("bad json")):
            with pytest.raises(HTTPException) as exc:
                svc.run_clarify(db, 51, course, "summarize", [])
        assert exc.value.status_code == 502
        assert _detail_code(exc) == "clarify_failed"
