"""Postgres checkpointer for the course-creation graph.

`PostgresSaver.from_conn_string` in langgraph-checkpoint-postgres 3.x is a
*context manager* that closes its connection on exit, so it cannot be used to
hand back a long-lived saver. We instead own a module-level connection pool and
construct the saver directly from it; the pool outlives any single graph run,
which is what a Celery worker (Task 10) needs.
"""

from __future__ import annotations

import atexit
from threading import Lock

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import settings

_lock = Lock()
_pool: ConnectionPool | None = None
_saver: PostgresSaver | None = None


def _psycopg_conninfo() -> str:
    """SQLAlchemy URLs carry a driver suffix that libpq does not understand."""
    return settings.database_url.replace("postgresql+psycopg://", "postgresql://")


def get_checkpointer() -> PostgresSaver:
    """Return the process-wide PostgresSaver, creating it on first use.

    `setup()` is idempotent: it creates the checkpoint tables if missing and
    applies any pending checkpoint-schema migrations.
    """
    global _pool, _saver
    with _lock:
        if _saver is None:
            _pool = ConnectionPool(
                conninfo=_psycopg_conninfo(),
                min_size=1,
                max_size=10,
                # PostgresSaver requires autocommit connections with dict rows,
                # and prepared statements must be off for pgbouncer-style pools.
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                },
                open=True,
            )
            saver = PostgresSaver(_pool)
            saver.setup()
            _saver = saver
            # Without this the pool's worker threads are still running at
            # interpreter shutdown and its __del__ raises PythonFinalizationError.
            atexit.register(_close_checkpointer)
        return _saver


def _close_checkpointer() -> None:
    global _pool, _saver
    with _lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _saver = None
