from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course
from app.models.export import ExportJob
from app.models.user import User
from app.tasks.export_tasks import run_export_task


def _seed_job(kind="course", status="pending"):
    with SessionLocal() as db:
        db.add(User(id=61, email="export-task@example.com", password_hash="x"))
        course = Course(topic_slug=f"export-task-{kind}", topic_raw="Task",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        job = ExportJob(course_id=course.id, user_id=61, kind=kind, status=status,
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def test_course_export_renders_uploads_and_completes():
    job_id = _seed_job("course")
    with patch("app.services.exports.gather_course_payload", return_value={"title": "Task"}) as mock_gather, \
        patch("app.documents.render.render_course_pdf", return_value=b"%PDF-course%") as mock_render, \
        patch("app.tasks.export_tasks.ensure_bucket") as mock_bucket, \
        patch("app.tasks.export_tasks.upload_object") as mock_upload:
        run_export_task(job_id)  # pyright: ignore[reportCallIssue]

    assert mock_gather.call_count == 1
    assert mock_render.call_count == 1
    mock_bucket.assert_called_once_with()
    mock_upload.assert_called_once()
    # upload_object(key, data, content_type) is called positionally.
    upload_args = mock_upload.call_args.args
    assert upload_args[2] == "application/pdf"
    assert upload_args[1] == b"%PDF-course%"
    with SessionLocal() as db:
        job = db.get(ExportJob, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.result_key is not None
        assert job.result_size == len(b"%PDF-course%")
        assert job.completed_at is not None
        assert job.error is None


def test_failed_export_marks_job_failed_with_generic_error():
    job_id = _seed_job("assignments")
    with patch("app.services.exports.gather_assignments_payload", side_effect=ValueError("boom")), \
        patch("app.tasks.export_tasks.ensure_bucket"), \
        patch("app.tasks.export_tasks.upload_object"):
        run_export_task(job_id)  # pyright: ignore[reportCallIssue]

    with SessionLocal() as db:
        job = db.get(ExportJob, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error == "PDF export failed. Please try again."
        assert job.result_key is None
        assert job.completed_at is not None


def test_completed_export_is_not_processed_again():
    job_id = _seed_job("full_course", status="succeeded")
    with patch("app.services.exports.gather_course_payload") as mock_gather:
        run_export_task(job_id)  # pyright: ignore[reportCallIssue]
    mock_gather.assert_not_called()


def test_custom_export_generates_markdown_and_pdf():
    job_id = _seed_job("custom")
    with SessionLocal() as db:
        job = db.get(ExportJob, job_id)
        assert job is not None
        job.params = {
            "brief": "Short summary",
            "plan": {
                "title": "Short summary",
                "output_kind": "summary",
                "length": "short",
                "item_count": None,
                "notes": None,
            },
        }
        db.commit()
    with patch("app.tasks.export_tasks.generate_custom_markdown", return_value="# Summary") as mock_generate, \
        patch("app.documents.render.render_custom_pdf", return_value=b"%PDF-custom%"), \
        patch("app.tasks.export_tasks.ensure_bucket"), \
        patch("app.tasks.export_tasks.upload_object") as mock_upload:
        run_export_task(job_id)  # pyright: ignore[reportCallIssue]

    mock_generate.assert_called_once()
    assert mock_upload.call_args.args[1] == b"%PDF-custom%"
    with SessionLocal() as db:
        job = db.get(ExportJob, job_id)
        assert job is not None
        assert job.status == "succeeded"
        assert job.result_key is not None
