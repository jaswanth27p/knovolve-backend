"""Does the Langfuse plumbing actually reach the code it is supposed to wrap?

Every other test in this suite exercises behavior that is identical whether or
not Langfuse is wired in, so deleting a `finally: flush_langfuse()` or
un-wrapping a `traced_workflow(...)` call would break the feature with the
whole suite still green. That gap is what this file closes:

* `flush_langfuse` must run on BOTH the success and the failure path of every
  Celery-scoped task wrapper. Celery worker children are recycled
  (celery_worker_max_tasks_per_child), and the OTel batch exporter's background
  thread gives no delivery guarantee before that happens — a missing flush
  silently drops the whole trace for that task.
* `traced_workflow` must actually be entered, with the right trace_name, at the
  workflow entry points, so one workflow is one trace.

The `flush_langfuse`/`traced_workflow` symbols are patched where they are
*used* (the task/service module that imported them), per this codebase's
convention, which is also what makes these tests fail if an import is dropped.
"""
import contextlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agents.course_creation.graph import CourseGenerationError
from app.db import SessionLocal
from app.models.course import Course, CourseJob
from app.models.course_extension import CourseExtensionJob
from app.models.export import ExportJob
from app.models.user import User
from app.tasks.assignment_tasks import (
    generate_chapter_assignment_task,
    generate_module_assignment_task,
    generate_module_topup_task,
)
from app.tasks.chapter_content_tasks import remediate_chapter_task
from app.tasks.course_creation_task import run_course_creation_job
from app.tasks.course_extension_task import run_course_extension_job
from app.tasks.evaluation_tasks import grade_assignment_attempt_task
from app.tasks.export_tasks import run_export_task


# --- fixtures / seed helpers -------------------------------------------------

def _make_user(user_id=901, email="langfuse-wiring@example.com"):
    with SessionLocal() as db:
        db.add(User(id=user_id, email=email, password_hash="x"))
        db.commit()
    return user_id


def _make_course(topic_slug="langfuse-wiring-course"):
    with SessionLocal() as db:
        course = Course(topic_slug=topic_slug, topic_raw="Langfuse Wiring",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        db.refresh(course)
        return course.id


def _make_course_job(topic_slug="langfuse-wiring-job"):
    with SessionLocal() as db:
        job = CourseJob(topic_slug=topic_slug, topic_raw="Langfuse Wiring Job",
                        topic_embedding=[1.0] + [0.0] * 2047, status="pending",
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def _make_extension_job(course_id, user_id):
    with SessionLocal() as db:
        job = CourseExtensionJob(course_id=course_id, user_id=user_id, request="add tcp",
                                 status="pending", created_at=datetime.now(timezone.utc),
                                 updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def _make_export_job(course_id, user_id, kind="course"):
    with SessionLocal() as db:
        job = ExportJob(course_id=course_id, user_id=user_id, kind=kind, status="pending",
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


@contextlib.contextmanager
def _recorded_trace(module_path: str):
    """Patch `traced_workflow` where `module_path` uses it and record the
    trace_name/kwargs each `with traced_workflow(...)` was opened with."""
    calls: list[tuple] = []

    def _fake(trace_name, **kwargs):
        calls.append((trace_name, kwargs))
        return contextlib.nullcontext()

    with patch(module_path, side_effect=_fake) as mock:
        mock.recorded = calls
        yield calls


# --- Celery-scoped flush wiring, success + failure path per module ----------

def test_course_creation_task_flushes_on_success():
    job_id = _make_course_job()
    course_id = _make_course()
    with patch("app.tasks.course_creation_task.flush_langfuse") as flush, \
            patch("app.tasks.course_creation_task.build_course_creation_graph") as build:
        build.return_value.invoke.return_value = {"error": None, "existing_course_id": course_id}
        run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()


def test_course_creation_task_flushes_on_failure():
    job_id = _make_course_job("langfuse-wiring-job-fail")
    with patch("app.tasks.course_creation_task.flush_langfuse") as flush, \
            patch("app.tasks.course_creation_task.build_course_creation_graph",
                  side_effect=CourseGenerationError("boom")):
        run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()
    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        assert job is not None and job.status == "failed"


def test_chapter_content_task_flushes_on_success():
    with patch("app.tasks.chapter_content_tasks.flush_langfuse") as flush, \
            patch("app.tasks.chapter_content_tasks.remediate_chapter") as remediate:
        remediate_chapter_task(1, 2, ["tag"], 3)  # pyright: ignore[reportCallIssue]
    remediate.assert_called_once()
    flush.assert_called_once_with()


def test_chapter_content_task_flushes_on_failure():
    with patch("app.tasks.chapter_content_tasks.flush_langfuse") as flush, \
            patch("app.tasks.chapter_content_tasks.remediate_chapter",
                  side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            remediate_chapter_task(1, 2, ["tag"], 3)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()


@pytest.mark.parametrize(
    ("task", "target", "args"),
    [
        (generate_chapter_assignment_task, "generate_chapter_assignment", (1,)),
        (generate_module_assignment_task, "generate_module_assignment", (1,)),
        (generate_module_topup_task, "generate_module_topup", (1, 2)),
    ],
)
def test_assignment_tasks_flush_on_success(task, target, args):
    """All three assignment entry points, not just one — each has its own
    `finally` block."""
    with patch("app.tasks.assignment_tasks.flush_langfuse") as flush, \
            patch(f"app.tasks.assignment_tasks.{target}") as inner:
        task(*args)  # pyright: ignore[reportCallIssue]
    inner.assert_called_once()
    flush.assert_called_once_with()


@pytest.mark.parametrize(
    ("task", "target", "args"),
    [
        (generate_chapter_assignment_task, "generate_chapter_assignment", (1,)),
        (generate_module_assignment_task, "generate_module_assignment", (1,)),
        (generate_module_topup_task, "generate_module_topup", (1, 2)),
    ],
)
def test_assignment_tasks_flush_on_failure(task, target, args):
    # ValueError is the "caller bug" branch of _retry_if_transient: it is
    # re-raised straight through rather than handed to Task.retry, so the
    # exception really does unwind past the `finally`.
    with patch("app.tasks.assignment_tasks.flush_langfuse") as flush, \
            patch(f"app.tasks.assignment_tasks.{target}", side_effect=ValueError("boom")):
        with pytest.raises(ValueError, match="boom"):
            task(*args)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()


def test_evaluation_task_flushes_on_success():
    with patch("app.tasks.evaluation_tasks.flush_langfuse") as flush, \
            patch("app.tasks.evaluation_tasks.grade_assignment_attempt") as grade:
        grade_assignment_attempt_task(7)  # pyright: ignore[reportCallIssue]
    grade.assert_called_once()
    flush.assert_called_once_with()


def test_evaluation_task_flushes_on_failure():
    with patch("app.tasks.evaluation_tasks.flush_langfuse") as flush, \
            patch("app.tasks.evaluation_tasks.grade_assignment_attempt",
                  side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            grade_assignment_attempt_task(7)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()


def test_course_extension_task_flushes_on_success():
    user_id = _make_user(902, "langfuse-wiring-ext@example.com")
    job_id = _make_extension_job(_make_course("langfuse-wiring-ext-course"), user_id)
    with patch("app.tasks.course_extension_task.flush_langfuse") as flush, \
            patch("app.tasks.course_extension_task.plan_new_chapters", return_value=[]):
        run_course_extension_job(job_id)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()


def test_course_extension_task_flushes_on_failure():
    user_id = _make_user(903, "langfuse-wiring-ext-fail@example.com")
    job_id = _make_extension_job(_make_course("langfuse-wiring-ext-fail-course"), user_id)
    with patch("app.tasks.course_extension_task.flush_langfuse") as flush, \
            patch("app.tasks.course_extension_task.plan_new_chapters",
                  side_effect=RuntimeError("boom")):
        run_course_extension_job(job_id)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()
    with SessionLocal() as db:
        job = db.get(CourseExtensionJob, job_id)
        assert job is not None and job.status == "failed"


def test_export_task_flushes_on_success():
    user_id = _make_user(904, "langfuse-wiring-export@example.com")
    job_id = _make_export_job(_make_course("langfuse-wiring-export-course"), user_id)
    with patch("app.tasks.export_tasks.flush_langfuse") as flush, \
            patch("app.services.exports.gather_course_payload", return_value={"title": "x"}), \
            patch("app.documents.render.render_course_pdf", return_value=b"%PDF%"), \
            patch("app.tasks.export_tasks.ensure_bucket"), \
            patch("app.tasks.export_tasks.upload_object"):
        run_export_task(job_id)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()


def test_export_task_flushes_on_failure():
    user_id = _make_user(905, "langfuse-wiring-export-fail@example.com")
    job_id = _make_export_job(_make_course("langfuse-wiring-export-fail-course"), user_id)
    with patch("app.tasks.export_tasks.flush_langfuse") as flush, \
            patch("app.services.exports.gather_course_payload", side_effect=RuntimeError("boom")), \
            patch("app.tasks.export_tasks.ensure_bucket"), \
            patch("app.tasks.export_tasks.upload_object"):
        run_export_task(job_id)  # pyright: ignore[reportCallIssue]
    flush.assert_called_once_with()
    with SessionLocal() as db:
        job = db.get(ExportJob, job_id)
        assert job is not None and job.status == "failed"


# --- traced_workflow wiring at a representative set of entry points ----------

def test_answer_chat_message_opens_the_chat_reply_trace():
    from app.services import chat

    from app.schemas.chat import LearnerContextBundle

    fresh_bundle = LearnerContextBundle(
        in_progress_count=0, completed_count=0, streak_current=0, my_courses=[],
    )
    with _recorded_trace("app.services.chat.traced_workflow") as calls, \
            patch("app.services.chat.build_context_bundle", return_value=fresh_bundle) as bundle, \
            patch("app.services.chat._build_agent") as build_agent:
        build_agent.return_value = (MagicMock(), [], MagicMock(content="hi"))
        req = MagicMock(context=None, current_route=None)
        chat.answer_chat_message(MagicMock(), 42, req)

    assert bundle.call_count == 1
    assert [name for name, _ in calls] == ["Chat Reply"]
    assert calls[0][1]["user_id"] == 42


def test_stream_chat_message_opens_the_chat_reply_trace():
    """The streaming entry point routes through traced_workflow_stream, so the
    trace is opened by langfuse_client, not by chat.py directly."""
    from app.services import chat

    def _fake_events(*_args, **_kwargs):
        yield {"type": "token", "text": "hi"}
        yield {"type": "done"}

    with _recorded_trace("app.llm.langfuse_client.traced_workflow") as calls, \
            patch("app.services.chat._stream_chat_message", side_effect=_fake_events):
        events = list(chat.stream_chat_message(MagicMock(), 42, MagicMock()))

    assert [e["type"] for e in events] == ["token", "done"]
    assert [name for name, _ in calls] == ["Chat Reply"]
    assert calls[0][1]["user_id"] == 42
    assert "streaming" in calls[0][1]["tags"]


def test_stream_chapter_content_opens_the_chapter_generation_trace():
    from app.agents.chapter_content import generate

    def _fake_events(*_args, **_kwargs):
        yield {"type": "section"}
        yield {"type": "done"}

    with _recorded_trace("app.llm.langfuse_client.traced_workflow") as calls, \
            patch("app.agents.chapter_content.generate._stream_chapter_content",
                  side_effect=_fake_events):
        chapter = SimpleNamespace(id=7)
        events = list(generate.stream_chapter_content(chapter, MagicMock(), 42))  # pyright: ignore[reportArgumentType]

    assert [e["type"] for e in events] == ["section", "done"]
    assert [name for name, _ in calls] == ["Chapter Content Generation"]
    assert calls[0][1]["session_id"] == 7
    assert calls[0][1]["user_id"] == 42


def test_course_creation_task_opens_the_course_creation_trace():
    """At least one Celery-scoped workflow: the trace must wrap the graph
    invocation, not merely exist somewhere in the module."""
    job_id = _make_course_job("langfuse-wiring-job-trace")
    course_id = _make_course("langfuse-wiring-trace-course")
    with _recorded_trace("app.tasks.course_creation_task.traced_workflow") as calls, \
            patch("app.tasks.course_creation_task.flush_langfuse"), \
            patch("app.tasks.course_creation_task.build_course_creation_graph") as build:
        build.return_value.invoke.return_value = {"error": None, "existing_course_id": course_id}
        run_course_creation_job(job_id)  # pyright: ignore[reportCallIssue]

    assert [name for name, _ in calls] == ["Course Creation"]
    assert calls[0][1]["session_id"] == job_id
    assert build.return_value.invoke.call_count == 1
