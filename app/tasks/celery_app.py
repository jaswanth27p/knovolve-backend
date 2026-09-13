import os

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init

from app.config import settings
from app.db import engine as _db_engine
from app.observability import instrument_static, setup_logging, setup_tracing
import app.models  # noqa: F401  (registers every model on Base.metadata before any task runs)


@worker_process_init.connect
def _reset_db_pool_after_fork(**_kwargs: object) -> None:
    """Drop the connection pool inherited across a prefork fork.

    The engine is built at import time in the parent, so every child inherits
    the same pooled sockets; two processes sharing one socket corrupts the
    wire protocol. close=False discards the child's pool without closing the
    parent's file descriptors, so the child lazily opens fresh connections.
    """
    _db_engine.dispose(close=False)

setup_tracing()
setup_logging()

celery_app = Celery("knovolve", broker=settings.redis_url, backend=settings.redis_url)

# Register the concrete task modules so the documented worker invocation
# (`celery -A app.tasks.celery_app worker --loglevel=info`) actually picks up
# the jobs users enqueue. Without this the worker connects but has zero tasks.
celery_app.conf.include = [
    "app.tasks.course_creation_task",
    "app.tasks.course_extension_task",
    "app.tasks.render_diagram_task",
    "app.tasks.assignment_tasks",
    "app.tasks.evaluation_tasks",
    "app.tasks.chapter_content_tasks",
    "app.tasks.auth_tasks",
    "app.tasks.export_tasks",
    "app.tasks.course_generation_task",
    "app.tasks.job_reconciliation_tasks",
]

# Periodic hygiene: revoked/expired refresh_tokens rows otherwise accumulate
# forever (every rotation - i.e. every /auth/refresh call - adds one and
# leaves its predecessor revoked). Runs once a day; the row is deleted only
# after settings.refresh_token_purge_after_days.
celery_app.conf.beat_schedule = {
    "purge-expired-refresh-tokens": {
        "task": "app.tasks.auth_tasks.purge_expired_refresh_tokens_task",
        "schedule": crontab(hour=3, minute=0),
    },
    # Backstop for jobs no per-task time limit can catch: one purged from the
    # queue or revoked before any worker picked it up. Runs often (user-facing
    # "is it done yet?" polling, unlike the once-a-day token purge above) so a
    # stuck job fails visibly within minutes, not indefinitely.
    "reconcile-stuck-generation-jobs": {
        "task": "app.tasks.job_reconciliation_tasks.reconcile_stuck_jobs_task",
        "schedule": 300.0,
    },
}

# Crash-resumability: ack the message only after the task body finishes, so a
# worker killed mid-run does not lose the job (the broker redelivers it once
# the Redis visibility timeout elapses instead of it zombifying as
# permanently "running"). visibility_timeout must exceed the longest single
# graph run to avoid double-execution while a healthy run is still going.
celery_app.conf.update(
    task_acks_late=True,
    task_time_limit=settings.celery_task_time_limit_seconds,
    task_soft_time_limit=settings.celery_task_soft_time_limit_seconds,
    # A worker killed outright (OOM, container restart, manual kill, or the
    # hard time limit above) would otherwise leave its message invisible for
    # the full visibility_timeout (6h) before redelivery — during which the
    # job's DB row just sits "running" with nothing to explain why. Rejecting
    # on worker-lost requeues it immediately instead. Every task here is
    # written to tolerate a redelivery safely (idempotent get-or-create,
    # status guards, or checkpoint-resume), so immediate requeue is safe.
    task_reject_on_worker_lost=True,
    broker_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout_seconds
    },
    # Bound concurrency and memory. `worker_concurrency` is the hard cap on
    # simultaneous tasks (prefork child processes); max-memory-per-child recycles
    # a child that bloats or leaks after a task, and max-tasks-per-child is a
    # blunter backstop. The memory/task limits are prefork-only and are ignored
    # by the solo pool. See AGENTS.md for the macOS prefork caveat.
    worker_concurrency=settings.celery_worker_concurrency,
    worker_prefetch_multiplier=1,
    worker_max_memory_per_child=settings.celery_worker_max_memory_per_child_kb,
    worker_max_tasks_per_child=settings.celery_worker_max_tasks_per_child,
)

# Tests run the task body synchronously (no worker, no broker round-trip) by
# setting CELERY_TASK_ALWAYS_EAGER=1 in the environment before import.
if os.environ.get("CELERY_TASK_ALWAYS_EAGER") == "1":
    celery_app.conf.task_always_eager = True

instrument_static()
