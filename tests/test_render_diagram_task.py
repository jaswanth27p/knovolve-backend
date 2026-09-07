from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.tasks.render_diagram_task import render_diagram_task, render_and_upload_diagram


def _make_pending_section(topic_slug: str) -> int:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=topic_slug, topic_raw=topic_slug, topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
        db.add(chapter)
        db.commit()
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="generating",
                                  outline=[], created_at=now, updated_at=now)
        db.add(content)
        db.commit()
        section = ChapterContentSection(
            chapter_content_id=content.id, order=0, heading="H", kind="teaching",
            body_markdown="body", examples=[{"prompt": "p", "walkthrough": "w"}],
            diagram_spec={"nodes": [{"id": "a", "label": "A"}], "edges": []},
            diagram_status="pending",
        )
        db.add(section)
        db.commit()
        return section.id


def test_render_and_upload_diagram_returns_public_url():
    section_id = _make_pending_section("diagram-task-helper")
    with SessionLocal() as db:
        section = db.get(ChapterContentSection, section_id)

    with patch("app.tasks.render_diagram_task.render_diagram_svg", return_value=b"<svg/>") as mock_render, \
         patch("app.tasks.render_diagram_task.ensure_bucket") as mock_ensure, \
         patch("app.tasks.render_diagram_task.upload_object") as mock_upload, \
         patch("app.tasks.render_diagram_task.get_public_url", return_value="http://x/images/1.svg"):
        url = render_and_upload_diagram(section)

    assert url == "http://x/images/1.svg"
    mock_render.assert_called_once_with(section.diagram_spec)
    mock_ensure.assert_called_once()
    mock_upload.assert_called_once_with(f"images/chapter-content-sections/{section.id}.svg", b"<svg/>", "image/svg+xml")


def test_render_diagram_task_success_marks_ready_and_publishes():
    section_id = _make_pending_section("diagram-task-success")

    with patch("app.tasks.render_diagram_task.render_and_upload_diagram", return_value="http://x/y.svg"), \
         patch("app.tasks.render_diagram_task.publish_event") as mock_publish:
        render_diagram_task(section_id)  # pyright: ignore[reportCallIssue]

    with SessionLocal() as db:
        section = db.get(ChapterContentSection, section_id)
        assert section is not None
        assert section.diagram_status == "ready"
        assert section.diagram_image_url == "http://x/y.svg"
    mock_publish.assert_called_once()
    event = mock_publish.call_args[0][1]
    assert event["type"] == "diagram_ready"
    assert event["diagram_image_url"] == "http://x/y.svg"


def test_render_diagram_task_skips_already_resolved_section():
    """A duplicate task delivery for an already-resolved section must not
    re-render or re-publish."""
    section_id = _make_pending_section("diagram-task-already-resolved")
    with SessionLocal() as db:
        section = db.get(ChapterContentSection, section_id)
        assert section is not None
        section.diagram_status = "ready"
        section.diagram_image_url = "http://already/there.svg"
        db.commit()

    with patch("app.tasks.render_diagram_task.render_and_upload_diagram") as mock_render, \
         patch("app.tasks.render_diagram_task.publish_event") as mock_publish:
        render_diagram_task(section_id)  # pyright: ignore[reportCallIssue]

    mock_render.assert_not_called()
    mock_publish.assert_not_called()


def test_render_diagram_task_transient_failure_increments_attempts():
    """Under a direct call, celery's self.retry() re-raises the original
    exception immediately (same behaviour documented in
    test_course_creation_task.py's transient-failure test) — this verifies
    the failed attempt is recorded before that happens."""
    section_id = _make_pending_section("diagram-task-transient")

    import pytest
    with patch("app.tasks.render_diagram_task.render_and_upload_diagram", side_effect=RuntimeError("dot crashed")):
        with pytest.raises(RuntimeError, match="dot crashed"):
            render_diagram_task(section_id)  # pyright: ignore[reportCallIssue]

    with SessionLocal() as db:
        section = db.get(ChapterContentSection, section_id)
        assert section is not None
        assert section.diagram_attempts == 1
        assert section.diagram_status == "pending"  # not yet given up


def test_render_diagram_task_exhausts_retries_marks_failed():
    """`.apply()` runs the task through celery's real retry machinery
    in-process (eager execution, no broker needed) so self.retry() actually
    loops and recurses until max_retries is exceeded — a direct call can't
    exercise this because it short-circuits retry() into an immediate
    re-raise (see the transient-failure test above)."""
    section_id = _make_pending_section("diagram-task-exhausted")

    with patch("app.tasks.render_diagram_task.render_and_upload_diagram", side_effect=RuntimeError("dot crashed")), \
         patch("app.tasks.render_diagram_task.publish_event") as mock_publish:
        result = render_diagram_task.apply(args=(section_id,))

    assert result.successful()  # the task itself completes normally after giving up
    with SessionLocal() as db:
        section = db.get(ChapterContentSection, section_id)
        assert section is not None
        assert section.diagram_status == "failed"
        assert section.diagram_attempts == 4  # 1 initial attempt + 3 retries
    mock_publish.assert_called_once()
    assert mock_publish.call_args[0][1]["type"] == "diagram_failed"
