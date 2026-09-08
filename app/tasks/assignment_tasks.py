from app.db import SessionLocal
from app.agents.assignment.generate import generate_chapter_assignment
from app.tasks.celery_app import celery_app


@celery_app.task
def generate_chapter_assignment_task(chapter_content_id: int) -> None:
    with SessionLocal() as db:
        generate_chapter_assignment(chapter_content_id, db)
