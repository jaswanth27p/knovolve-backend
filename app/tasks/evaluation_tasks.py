from app.db import SessionLocal
from app.agents.evaluation.grade import grade_assignment_attempt
from app.tasks.celery_app import celery_app


@celery_app.task
def grade_assignment_attempt_task(attempt_id: int) -> None:
    with SessionLocal() as db:
        grade_assignment_attempt(attempt_id, db)
