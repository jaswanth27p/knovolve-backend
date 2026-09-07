import os

from celery import Celery

from app.config import settings

celery_app = Celery("knovolve", broker=settings.redis_url, backend=settings.redis_url)

# Crash-resumability: ack the message only after the task body finishes, so a
# worker killed mid-run does not lose the job (the broker redelivers it once
# the Redis visibility timeout elapses instead of it zombifying as
# permanently "running"). visibility_timeout must exceed the longest single
# graph run to avoid double-execution while a healthy run is still going.
celery_app.conf.update(
    task_acks_late=True,
    broker_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout_seconds
    },
)

# Tests run the task body synchronously (no worker, no broker round-trip) by
# setting CELERY_TASK_ALWAYS_EAGER=1 in the environment before import.
if os.environ.get("CELERY_TASK_ALWAYS_EAGER") == "1":
    celery_app.conf.task_always_eager = True
