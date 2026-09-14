"""One-time LangGraph checkpoint schema setup for the course-creation graph.

Must run from a clean process (no other open transaction) before any Celery
worker builds the course-creation graph — see
``app/agents/course_creation/checkpointer.py`` for why. Run as part of the
one-shot migration step, alongside ``alembic upgrade head``.

Usage: `python -m scripts.setup_checkpointer`
"""

from __future__ import annotations

from app.agents.course_creation.checkpointer import ensure_schema

if __name__ == "__main__":
    ensure_schema()
    print("checkpoint schema ready")
