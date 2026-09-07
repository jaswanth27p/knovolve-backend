import os

from celery import Celery

from app.config import settings

celery_app = Celery("knovolve", broker=settings.redis_url, backend=settings.redis_url)

# Tests run the task body synchronously (no worker, no broker round-trip) by
# setting CELERY_TASK_ALWAYS_EAGER=1 in the environment before import.
if os.environ.get("CELERY_TASK_ALWAYS_EAGER") == "1":
    celery_app.conf.task_always_eager = True
