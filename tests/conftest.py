import pytest
from sqlalchemy import text
from app.config import settings
from app.db import SessionLocal
# Import all models to register them with SQLAlchemy's Base metadata
from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.models.course import Course, CourseJob, Module, Chapter, Concept, ConceptEdge
from app.models.course_extension import CourseExtensionJob
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.enrollment import UserCourse
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt, AssignmentAnswer
from app.models.learner_streak import LearnerStreak
from app.models.export import CourseGenerationRun, ExportJob


def _database_name(url: str) -> str:
    tail = url.rsplit("/", 1)[-1]
    return tail.split("?", 1)[0]


@pytest.fixture(scope="session", autouse=True)
def _refuse_non_test_database():
    """Hard safety rail: `clean_db` TRUNCATEs every table after each test, so
    running the suite against the dev database destroys the developer's data.
    Refuse to run unless the target database name ends in `_test`.

    Run the suite with an explicit test database, e.g.:
      DATABASE_URL=postgresql+psycopg://knovolve:knovolve@localhost:5432/knovolve_test \\
        .venv/bin/pytest -q
    """
    name = _database_name(settings.database_url)
    if not name.endswith("_test"):
        pytest.exit(
            f"Refusing to run tests against database {name!r}: this suite "
            "TRUNCATEs all tables. Set DATABASE_URL to a *_test database.",
            returncode=2,
        )
    yield



class _NullRedis:
    """Stands in for Redis in tests: reads miss, writes succeed silently.
    Course preview caching is convenience-only (missing entries degrade to
    recomputation), so a null client keeps the suite hermetic."""
    def get(self, _):
        return None

    def set(self, *_args, **_kwargs):
        return True

    def delete(self, *_args):
        return 1


@pytest.fixture(autouse=True)
def _no_redis(monkeypatch):
    from app.services import course_preview
    monkeypatch.setattr(course_preview, "_get_client", lambda: _NullRedis())


@pytest.fixture(autouse=True)
def _web_search_off(monkeypatch):
    """Keep the suite hermetic: all existing course-creation tests exercise the
    flag-off path. Tests for the web path re-enable it explicitly."""
    from app.config import settings
    monkeypatch.setattr(settings, "web_search_enabled", False)


@pytest.fixture(autouse=True)
def _chapter_research_off(monkeypatch):
    """Keep the suite hermetic: chapter-content research would otherwise make
    real web calls. Tests for it patch the same symbol themselves."""
    from app.agents.chapter_content import research as chapter_research
    monkeypatch.setattr(chapter_research, "fetch_chapter_research", lambda *args, **kwargs: "")


@pytest.fixture(autouse=True)
def clean_db():
    yield
    with SessionLocal() as s:
        # Truncate whatever exists rather than a hardcoded list: the LangGraph
        # checkpoint tables are created by PostgresSaver.setup() at runtime (not
        # by migrations), so a freshly-migrated test DB won't have them. Never
        # touch alembic_version.
        tables = [
            row[0]
            for row in s.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            if row[0] != "alembic_version"
        ]
        if not tables:
            return
        quoted = ", ".join(f'"{name}"' for name in tables)
        s.execute(text(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE"))
        s.commit()
