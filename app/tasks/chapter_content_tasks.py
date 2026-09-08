from app.db import SessionLocal
from app.agents.chapter_content.remediate import remediate_chapter
from app.tasks.celery_app import celery_app


@celery_app.task
def remediate_chapter_task(chapter_id: int, user_id: int, weak_concept_tags: list[str], source_attempt_id: int) -> None:
    with SessionLocal() as db:
        remediate_chapter(chapter_id, user_id, weak_concept_tags, source_attempt_id, db)
