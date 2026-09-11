from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from app.db import SessionLocal
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Course, Module, Chapter
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.user import User
from app.agents.generation import ensure as ensure_module


def _now():
    return datetime.now(timezone.utc)


def _seed_chapter(user_id=71, slug="generation-ensure"):
    """Seed a global module/chapter plus a graded module-level attempt for
    `user_id`. The attempt is only a FK target for remediation rows (the
    real schema constrains remediation_source_attempt_id -> assignment_attempts).
    Returns (chapter_id, attempt_id)."""
    with SessionLocal() as db:
        db.add(User(id=user_id, email=f"g{user_id}@example.com", password_hash="x"))
        course = Course(topic_slug=slug, topic_raw="Ensure",
                        topic_embedding=[0.0] * 2048, created_at=datetime.now(timezone.utc))
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1, scope="global")
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1, scope="global")
        db.add(chapter)
        db.commit()
        assignment = Assignment(level="module", module_id=module.id, scope="global", status="ready",
                                created_at=_now(), updated_at=_now())
        db.add(assignment)
        db.commit()
        attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user_id, status="graded",
                                    overall_score=0.0, created_at=_now(), updated_at=_now())
        db.add(attempt)
        db.commit()
        return chapter.id, attempt.id


def test_ensure_chapter_content_generates_missing_sections_only():
    chapter_id, _ = _seed_chapter()
    outline = [MagicMock(heading="One", objective="o", kind="teaching"),
               MagicMock(heading="Two", objective="o", kind="teaching")]
    section_result = MagicMock()
    section_result.body_markdown = "body"
    section_result.examples = []
    section_result.diagram_spec = None
    for draft in outline:
        draft.model_dump.return_value = {"heading": draft.heading, "objective": "o", "kind": "teaching"}

    with SessionLocal() as db:
        from app.models.course import Chapter as ChapterModel
        chapter = db.get(ChapterModel, chapter_id)
        assert chapter is not None
        with patch("app.agents.generation.ensure.generate_section_outline", return_value=outline), \
            patch("app.agents.generation.ensure.generate_chapter_section", return_value=section_result):
            ensure_module.ensure_chapter_content(db, chapter)
            ensure_module.ensure_chapter_content(db, chapter)

        contents = db.query(ChapterContent).filter_by(chapter_id=chapter_id).all()
        assert len(contents) == 1
        assert contents[0].status == "ready"
        sections = db.query(ChapterContentSection).filter_by(chapter_content_id=contents[0].id).all()
        assert len(sections) == 2


def test_ensure_chapter_content_ignores_remediation_rows():
    chapter_id, attempt_id = _seed_chapter(user_id=72, slug="generation-remediation-guard")
    with SessionLocal() as db:
        from app.models.course import Chapter as ChapterModel
        chapter = db.get(ChapterModel, chapter_id)
        assert chapter is not None
        from app.models.chapter_content import ChapterContent as Content
        now = datetime.now(timezone.utc)
        # A base row that still needs generating, plus a failed remediation row
        # for the same chapter. `ensure_chapter_content` must generate only the
        # base and never author into the remediation row.
        base = Content(chapter_id=chapter_id, version=1, scope="global", status="failed",
                       outline=[], created_at=now, updated_at=now)
        remediation = Content(chapter_id=chapter_id, version=2, scope="user", user_id=72,
                              status="failed", outline=[], remediation_source_attempt_id=attempt_id,
                              created_at=now, updated_at=now)
        db.add_all([base, remediation])
        db.commit()
        base_id, remediation_id = base.id, remediation.id
        with patch("app.agents.generation.ensure.generate_section_outline",
                   return_value=[]) as mock_outline:
            ensure_module.ensure_chapter_content(db, chapter)
        mock_outline.assert_called_once()
        remediation = db.get(Content, remediation_id)
        assert remediation is not None
        assert remediation.status == "failed"
        assert db.query(ChapterContentSection).filter_by(
            chapter_content_id=remediation_id
        ).count() == 0
        base = db.get(Content, base_id)
        assert base is not None
        assert base.status == "ready"


def test_failed_diagram_does_not_block_ready_chapter():
    chapter_id, _ = _seed_chapter(user_id=73, slug="generation-diagram")
    outline = [MagicMock(heading="One", objective="o", kind="teaching")]
    outline[0].model_dump.return_value = {"heading": "One", "objective": "o", "kind": "teaching"}
    section_result = MagicMock()
    section_result.body_markdown = "body"
    section_result.examples = []
    section_result.diagram_spec = {"nodes": [], "edges": []}
    with SessionLocal() as db:
        from app.models.course import Chapter as ChapterModel
        chapter = db.get(ChapterModel, chapter_id)
        assert chapter is not None
        with patch("app.agents.generation.ensure.generate_section_outline", return_value=outline), \
            patch("app.agents.generation.ensure.generate_chapter_section", return_value=section_result), \
            patch("app.agents.generation.ensure.render_and_upload_diagram", side_effect=ValueError("bad spec")):
            ensure_module.ensure_chapter_content(db, chapter)
        content = db.query(ChapterContent).filter_by(chapter_id=chapter_id).one()
        section = db.query(ChapterContentSection).filter_by(chapter_content_id=content.id).one()
        assert content.status == "ready"
        assert section.diagram_status == "failed"


def test_ensure_chapter_content_finalizes_pending_diagram_on_existing_sections():
    chapter_id, _ = _seed_chapter(user_id=76, slug="generation-resume-diagram")
    with SessionLocal() as db:
        now = _now()
        content = ChapterContent(
            chapter_id=chapter_id, version=1, scope="global", status="generating",
            outline=[{"heading": "One", "objective": "o", "kind": "teaching"}],
            created_at=now, updated_at=now,
        )
        db.add(content)
        db.commit()
        section = ChapterContentSection(
            chapter_content_id=content.id, order=0, heading="One", kind="teaching",
            body_markdown="body", examples=[],
            diagram_spec={"nodes": [], "edges": []}, diagram_status="pending",
        )
        db.add(section)
        db.commit()
        content_id, section_id = content.id, section.id

        from app.models.course import Chapter as ChapterModel
        chapter = db.get(ChapterModel, chapter_id)
        assert chapter is not None
        with patch("app.agents.generation.ensure.render_and_upload_diagram",
                   return_value="https://example.com/diagram.png"):
            ensure_module.ensure_chapter_content(db, chapter)
            content = db.get(ChapterContent, content_id)
            assert content is not None
            assert content.status == "ready"
            section = db.get(ChapterContentSection, section_id)
            assert section is not None
            assert section.diagram_status == "ready"
            assert section.diagram_image_url == "https://example.com/diagram.png"

            ensure_module.ensure_chapter_content(db, chapter)
            section = db.get(ChapterContentSection, section_id)
            assert section is not None
            assert section.diagram_status == "ready"
            assert section.diagram_image_url == "https://example.com/diagram.png"


def test_ensure_chapter_assignment_requires_ready_and_calls_generator():
    chapter_id, _ = _seed_chapter(user_id=74, slug="generation-assignment")
    with SessionLocal() as db:
        now = _now()
        content = ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="ready",
                                 outline=[], created_at=now, updated_at=now)
        db.add(content)
        db.commit()

        with patch("app.agents.generation.ensure.generate_chapter_assignment") as mock_gen, \
            patch("app.agents.generation.ensure._chapter_assignment",
                  return_value=MagicMock(status="ready")):
            ensure_module.ensure_chapter_assignment(db, content)
        mock_gen.assert_called_once_with(content.id, db)

        content.status = "generating"
        db.commit()
        with pytest.raises(ValueError):
            ensure_module.ensure_chapter_assignment(db, content)


def _seed_ready_chapter_content(user_id, slug):
    chapter_id, _ = _seed_chapter(user_id=user_id, slug=slug)
    with SessionLocal() as db:
        now = _now()
        content = ChapterContent(chapter_id=chapter_id, version=1, scope="global", status="ready",
                                 outline=[], created_at=now, updated_at=now)
        db.add(content)
        db.commit()
        return content.id


def test_ensure_chapter_assignment_raises_when_generated_assignment_failed():
    content_id = _seed_ready_chapter_content(77, "generation-assignment-failed")
    with SessionLocal() as db:
        now = _now()
        db.add(Assignment(level="chapter", scope="global", chapter_content_id=content_id,
                          status="failed", created_at=now, updated_at=now))
        db.commit()
        content = db.get(ChapterContent, content_id)
        assert content is not None
        with patch("app.agents.generation.ensure.generate_chapter_assignment"), \
            pytest.raises(ValueError, match="Chapter assignment generation failed"):
            ensure_module.ensure_chapter_assignment(db, content)


def test_ensure_chapter_assignment_raises_when_no_assignment_persisted():
    content_id = _seed_ready_chapter_content(78, "generation-assignment-missing")
    with SessionLocal() as db:
        content = db.get(ChapterContent, content_id)
        assert content is not None
        with patch("app.agents.generation.ensure.generate_chapter_assignment"), \
            pytest.raises(ValueError, match="Chapter assignment generation failed"):
            ensure_module.ensure_chapter_assignment(db, content)


@pytest.mark.parametrize("status", ["ready", "generating"])
def test_ensure_chapter_assignment_allows_non_failed_status(status):
    content_id = _seed_ready_chapter_content(79, f"generation-assignment-{status}")
    with SessionLocal() as db:
        now = _now()
        db.add(Assignment(level="chapter", scope="global", chapter_content_id=content_id,
                          status=status, created_at=now, updated_at=now))
        db.commit()
        content = db.get(ChapterContent, content_id)
        assert content is not None
        with patch("app.agents.generation.ensure.generate_chapter_assignment"):
            ensure_module.ensure_chapter_assignment(db, content)


def test_ensure_module_assignment_raises_when_generated_assignment_failed():
    _seed_chapter(user_id=80, slug="generation-module-assignment-failed")
    with SessionLocal() as db:
        assignment = db.query(Assignment).filter_by(level="module").one()
        assignment.status = "failed"
        db.commit()
        module_id = assignment.module_id
        assert module_id is not None
        with patch("app.agents.generation.ensure.generate_module_assignment"), \
            pytest.raises(ValueError, match="Module assignment generation failed"):
            ensure_module.ensure_module_assignment(db, module_id)


def test_ensure_module_assignment_raises_when_no_assignment_persisted():
    with SessionLocal() as db:
        with patch("app.agents.generation.ensure.generate_module_assignment"), \
            pytest.raises(ValueError, match="Module assignment generation failed"):
            ensure_module.ensure_module_assignment(db, 987654)


@pytest.mark.parametrize("status", ["ready", "generating"])
def test_ensure_module_assignment_allows_non_failed_status(status):
    _seed_chapter(user_id=81, slug=f"generation-module-assignment-{status}")
    with SessionLocal() as db:
        assignment = db.query(Assignment).filter_by(level="module").one()
        assignment.status = status
        db.commit()
        module_id = assignment.module_id
        assert module_id is not None
        with patch("app.agents.generation.ensure.generate_module_assignment"):
            ensure_module.ensure_module_assignment(db, module_id)


def test_ensure_remediation_content_guards_and_dispatches():
    chapter_id, attempt_id = _seed_chapter(user_id=75, slug="generation-remediation-dispatch")
    with SessionLocal() as db:
        now = _now()
        content = ChapterContent(chapter_id=chapter_id, version=2, scope="user", user_id=75,
                                 status="generating", outline=[], remediation_target_tags=["t"],
                                 remediation_source_attempt_id=attempt_id,
                                 created_at=now, updated_at=now)
        db.add(content)
        db.commit()

        with patch("app.agents.generation.ensure.remediate_chapter") as mock_remediate:
            ensure_module.ensure_remediation_content(db, chapter_id, 75, content)
        mock_remediate.assert_called_once_with(chapter_id, 75, ["t"], attempt_id, db)

        with pytest.raises(ValueError):
            ensure_module.ensure_remediation_content(db, chapter_id, 999, content)

        plain = ChapterContent(chapter_id=chapter_id, version=3, scope="user", user_id=75,
                               status="generating", outline=[], created_at=now, updated_at=now)
        db.add(plain)
        db.commit()
        with pytest.raises(ValueError):
            ensure_module.ensure_remediation_content(db, chapter_id, 75, plain)


def test_ensure_module_assignment_calls_generator():
    with SessionLocal() as db:
        with patch("app.agents.generation.ensure.generate_module_assignment") as mock_gen, \
            patch("app.agents.generation.ensure._module_assignment",
                  return_value=MagicMock(status="ready")):
            ensure_module.ensure_module_assignment(db, 42)
        mock_gen.assert_called_once_with(42, db)
