from app.db import SessionLocal
from app.agents.assignment.generate import (
    generate_chapter_assignment,
    generate_module_assignment,
    generate_module_topup,
)
from app.tasks.celery_app import celery_app


@celery_app.task
def generate_chapter_assignment_task(chapter_content_id: int) -> None:
    with SessionLocal() as db:
        generate_chapter_assignment(chapter_content_id, db)


@celery_app.task
def generate_module_assignment_task(module_id: int) -> None:
    with SessionLocal() as db:
        generate_module_assignment(module_id, db)


@celery_app.task
def generate_module_topup_task(assignment_id: int, user_id: int) -> None:
    with SessionLocal() as db:
        generate_module_topup(assignment_id, user_id, db)
