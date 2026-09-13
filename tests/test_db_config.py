from unittest.mock import patch

from sqlalchemy import Table
from sqlalchemy.pool import QueuePool


def test_engine_enables_pool_pre_ping_and_bounded_pool():
    from app.db import engine
    from app.config import settings

    pool = engine.pool
    assert isinstance(pool, QueuePool)
    assert getattr(pool, "_pre_ping") is True
    assert pool.size() == settings.db_pool_size


def test_worker_process_init_disposes_inherited_pool():
    import app.tasks.celery_app as celery_module

    with patch.object(celery_module._db_engine, "dispose") as dispose:
        celery_module._reset_db_pool_after_fork()
    dispose.assert_called_once_with(close=False)


def test_assignment_attempt_has_hot_path_composite_index():
    from app.models.attempt import AssignmentAttempt

    table = AssignmentAttempt.__table__
    assert isinstance(table, Table)
    names = {index.name for index in table.indexes}
    assert "ix_assignment_attempts_assignment_user_status_created" in names
