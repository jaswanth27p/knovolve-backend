"""Postgres checkpointer for the course-creation graph.

`PostgresSaver.from_conn_string` in langgraph-checkpoint-postgres 3.x is a
*context manager* that closes its connection on exit, so it cannot be used to
hand back a long-lived saver. We instead own a module-level connection pool and
construct the saver directly from it; the pool outlives any single graph run,
which is what a Celery worker (Task 10) needs.

Schema setup (``PostgresSaver.setup()``) is deliberately NOT done here anymore.
``setup()`` runs ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` on the checkpoint
tables, which must wait for every other session's open transaction to finish —
including the calling task's own outer DB session. Calling it lazily from
inside a Celery task self-deadlocked in production: the task's own session
(kept open across the whole run) blocked the index build, which the graph
build itself was waiting on before doing any work. ``ensure_schema()`` below
runs once, from the one-shot migration container, in a process with no other
open transaction, before any worker task ever calls :func:`get_checkpointer`.
"""

from __future__ import annotations

import atexit
from threading import Lock

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from app.config import settings

_lock = Lock()
_pool: ConnectionPool[Connection[DictRow]] | None = None
_saver: PostgresSaver | None = None


def _psycopg_conninfo() -> str:
    """SQLAlchemy URLs carry a driver suffix that libpq does not understand."""
    return settings.database_url.replace("postgresql+psycopg://", "postgresql://")


def _pool_kwargs() -> dict:
    return {
        "conninfo": _psycopg_conninfo(),
        # Declares the pool's connection type as Connection[DictRow] to
        # match PostgresSaver's expected `Conn` type. This only affects
        # static typing — the actual dict-row behavior at runtime comes
        # from `row_factory=dict_row` in kwargs below, since psycopg
        # does not derive row_factory from the generic parameter.
        "connection_class": Connection[DictRow],
        "min_size": 1,
        "max_size": 10,
        # PostgresSaver requires autocommit connections with dict rows,
        # and prepared statements must be off for pgbouncer-style pools.
        "kwargs": {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        "open": True,
    }


def get_checkpointer() -> PostgresSaver:
    """Return the process-wide PostgresSaver, creating it on first use.

    Assumes the checkpoint schema already exists — see :func:`ensure_schema`,
    which must run (via the migration container) before any worker calls this.
    """
    global _pool, _saver
    with _lock:
        if _saver is None:
            _pool = ConnectionPool(**_pool_kwargs())
            _saver = PostgresSaver(_pool)
            # Without this the pool's worker threads are still running at
            # interpreter shutdown and its __del__ raises PythonFinalizationError.
            atexit.register(_close_checkpointer)
        return _saver


def ensure_schema() -> None:
    """Create/upgrade the checkpoint tables. Run once, from a clean process
    with no other open transaction (the ``knovolve-migrate`` container), never
    from a worker — see the module docstring for why."""
    pool = ConnectionPool(**_pool_kwargs())
    try:
        PostgresSaver(pool).setup()
    finally:
        pool.close()


def _close_checkpointer() -> None:
    global _pool, _saver
    with _lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _saver = None
