import os

from celery import Celery

from app.config import settings
from app.observability import instrument_static, setup_logging, setup_tracing
import app.models  # noqa: F401  (registers every model on Base.metadata before any task runs)

setup_tracing()
setup_logging()

celery_app = Celery("knovolve", broker=settings.redis_url, backend=settings.redis_url)

# Register the concrete task modules so the documented worker invocation
# (`celery -A app.tasks.celery_app worker --loglevel=info`) actually picks up
# the jobs users enqueue. Without this the worker connects but has zero tasks.
celery_app.conf.include = [
    "app.tasks.course_creation_task",
    "app.tasks.render_diagram_task",
    "app.tasks.assignment_tasks",
    "app.tasks.evaluation_tasks",
]

# Crash-resumability: ack the message only after the task body finishes, so a
# worker killed mid-run does not lose the job (the broker redelivers it once
# the Redis visibility timeout elapses instead of it zombifying as
# permanently "running"). visibility_timeout must exceed the longest single
# graph run to avoid double-execution while a healthy run is still going.
celery_app.conf.update(
    task_acks_late=True,
    task_time_limit=settings.celery_task_time_limit_seconds,
    broker_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout_seconds
    },
)

# Tests run the task body synchronously (no worker, no broker round-trip) by
# setting CELERY_TASK_ALWAYS_EAGER=1 in the environment before import.
if os.environ.get("CELERY_TASK_ALWAYS_EAGER") == "1":
    celery_app.conf.task_always_eager = True

instrument_static()
