from datetime import datetime, timezone

from celery import Task

from app.agents.course_creation.graph import build_course_creation_graph
from app.agents.course_creation.state import CourseCreationState
from app.db import SessionLocal
from app.models.course import CourseJob
from app.tasks.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3, default_retry_delay=5)
def run_course_creation_job(self: Task, job_id: int) -> None:
    with SessionLocal() as db:
        job = db.get(CourseJob, job_id)
        if job is None:
            # `assert` is stripped under -O/PYTHONOPTIMIZE, so a missing row
            # (a genuine caller bug — Task 11 only enqueues ids it just
            # inserted) must be an explicit, non-optimizable failure rather
            # than a debug-mode-only invariant.
            raise ValueError(f"CourseJob {job_id} not found")

        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        db.commit()

        initial_state: CourseCreationState = {
            "job_id": job_id,
            "topic_raw": job.topic_raw,
            "topic_slug": job.topic_slug,
            "topic_embedding": job.topic_embedding,
            "existing_course_id": None,
            "modules": None,
            "concepts": None,
            "concept_edges": None,
            "error": None,
        }

        try:
            graph = build_course_creation_graph()
            final_state = graph.invoke(
                initial_state, config={"configurable": {"thread_id": str(job_id)}}
            )
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
            return

        job.status = "succeeded"
        job.course_id = final_state["existing_course_id"]
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
