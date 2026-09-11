"""Celery worker that converts approved export jobs into stored PDFs."""
import logging
from datetime import datetime, timezone

from celery import Task

from app.agents.custom_export.generate import generate_custom_markdown
from app.db import SessionLocal
from app.documents import render
from app.documents.render import markdown_to_html
from app.models.course import Course
from app.models.export import ExportJob
from app.services import exports as export_service
from app.storage.s3 import ensure_bucket, upload_object
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

GENERIC_EXPORT_ERROR = "PDF export failed. Please try again."


@celery_app.task(bind=True)
def run_export_task(self: Task, export_id: int) -> None:
    with SessionLocal() as db:
        job = db.get(ExportJob, export_id)
        if job is None:
            raise ValueError(f"ExportJob {export_id} not found")
        if job.status in ("succeeded", "failed"):
            return
        course = db.get(Course, job.course_id)
        if course is None:
            raise ValueError(f"Course {job.course_id} not found")
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
        try:
            if job.kind in ("course", "full_course"):
                payload = export_service.gather_course_payload(
                    db, course, job.user_id, full=(job.kind == "full_course")
                )
                pdf_bytes = render.render_course_pdf(payload)
            elif job.kind in ("assignments", "full_assignments"):
                payload = export_service.gather_assignments_payload(
                    db, course, job.user_id, full=(job.kind == "full_assignments")
                )
                pdf_bytes = render.render_assignments_pdf(payload)
            elif job.kind == "custom":
                custom_params = job.params or {}
                custom_markdown = generate_custom_markdown(
                    db, course, job.user_id, custom_params["brief"], custom_params["plan"]
                )
                payload = {
                    "title": custom_params["plan"]["title"],
                    "subtitle": "Custom export",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "body_html": markdown_to_html(custom_markdown),
                }
                pdf_bytes = render.render_custom_pdf(payload)
            else:
                raise ValueError(f"Unsupported export kind: {job.kind}")
            key = f"exports/{course.topic_slug}/{job.id}.pdf"
            ensure_bucket()
            upload_object(key, pdf_bytes, "application/pdf")
            job.result_key = key
            job.result_size = len(pdf_bytes)
            job.status = "succeeded"
            job.error = None
            job.completed_at = datetime.now(timezone.utc)
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.error("export job %s failed", export_id, exc_info=exc)
            job.status = "failed"
            job.error = GENERIC_EXPORT_ERROR
            job.completed_at = datetime.now(timezone.utc)
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
