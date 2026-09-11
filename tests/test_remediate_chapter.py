from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt
from app.models.user import User
from app.agents.chapter_content.nodes.generate_section_outline import SectionOutlineDraft
from app.agents.chapter_content.nodes.generate_chapter_section import (
    ChapterSectionResponse, ExampleDraft, DiagramSpecDraft, DiagramNodeDraft,
)
from app.agents.chapter_content.remediate import remediate_chapter


def _now():
    return datetime.now(timezone.utc)


def _make_chapter_with_v1_and_attempt(db, slug: str) -> tuple[int, int, int]:
    """Global V1 chapter content + one graded attempt against it (used only
    as the remediation_source_attempt_id FK target). Returns (chapter_id, attempt_id)."""
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
    db.add(course)
    db.flush()
    module = Module(course_id=course.id, title="M", objective="o", order=1)
    db.add(module)
    db.flush()
    chapter = Chapter(module_id=module.id, title="Recursion", objective="Understand recursive functions", order=1)
    db.add(chapter)
    db.flush()
    content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                              outline=[], created_at=_now(), updated_at=_now())
    db.add(content)
    db.flush()
    assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                             status="ready", created_at=_now(), updated_at=_now())
    db.add(assignment)
    db.flush()
    question = AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q",
                                   options=["a", "b"], correct_answer="a", explanation="e",
                                   concept_tag="recursion-base-case", difficulty="easy")
    db.add(question)
    db.flush()
    user = User(email=f"{slug}@example.com", password_hash="x")
    db.add(user)
    db.flush()
    attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                 overall_score=0.5, created_at=_now(), updated_at=_now())
    db.add(attempt)
    db.commit()
    return chapter.id, attempt.id, user.id


_OUTLINE = [
    SectionOutlineDraft(heading="Base cases", objective="Identify a base case", kind="teaching", order=1),
]
_SECTION_RESPONSE = ChapterSectionResponse(
    body_markdown="content", examples=[ExampleDraft(prompt="p", walkthrough="w")],
    diagram_spec=DiagramSpecDraft(nodes=[DiagramNodeDraft(id="a", label="A")], edges=[]),
)


def test_creates_new_user_scoped_version_continuing_chapter_numbering():
    with SessionLocal() as db:
        chapter_id, attempt_id, user_id = _make_chapter_with_v1_and_attempt(db, "remediate-a")

    with patch("app.agents.chapter_content.remediate.generate_remediation_outline", return_value=_OUTLINE), \
         patch("app.agents.chapter_content.remediate.generate_chapter_section", return_value=_SECTION_RESPONSE), \
         patch("app.agents.chapter_content.remediate.generate_chapter_assignment_task") as mock_assignment_task:
        with SessionLocal() as db:
            remediate_chapter(chapter_id, user_id, ["recursion-base-case"], attempt_id, db)

    with SessionLocal() as db:
        content = db.query(ChapterContent).filter_by(
            chapter_id=chapter_id, scope="user", user_id=user_id,
        ).one()
        assert content.version == 2  # continues global V1's numbering
        assert content.status == "ready"
        assert content.remediation_target_tags == ["recursion-base-case"]
        assert content.remediation_source_attempt_id == attempt_id
        sections = db.query(ChapterContentSection).filter_by(chapter_content_id=content.id).all()
        assert len(sections) == 1
        assert sections[0].heading == "Base cases"
        # No diagram generation for remediation sections in V1 — the diagram
        # the mocked generate_chapter_section returned must be dropped.
        assert sections[0].diagram_spec is None
        assert sections[0].diagram_status is None

    mock_assignment_task.delay.assert_called_once_with(content.id)


def test_second_remediation_for_same_user_continues_version_numbering():
    with SessionLocal() as db:
        chapter_id, attempt_id, user_id = _make_chapter_with_v1_and_attempt(db, "remediate-b")

    with patch("app.agents.chapter_content.remediate.generate_remediation_outline", return_value=_OUTLINE), \
         patch("app.agents.chapter_content.remediate.generate_chapter_section", return_value=_SECTION_RESPONSE), \
         patch("app.agents.chapter_content.remediate.generate_chapter_assignment_task"):
        with SessionLocal() as db:
            remediate_chapter(chapter_id, user_id, ["recursion-base-case"], attempt_id, db)

        # A second, DIFFERENT triggering attempt for the same user+chapter.
        with SessionLocal() as db:
            v2_content = db.query(ChapterContent).filter_by(chapter_id=chapter_id, scope="user", user_id=user_id).one()
            v2_assignment = Assignment(level="chapter", chapter_content_id=v2_content.id, scope="user", user_id=user_id,
                                        status="ready", created_at=_now(), updated_at=_now())
            db.add(v2_assignment)
            db.flush()
            attempt2 = AssignmentAttempt(assignment_id=v2_assignment.id, user_id=user_id, status="graded",
                                          overall_score=0.5, created_at=_now(), updated_at=_now())
            db.add(attempt2)
            db.commit()
            attempt2_id = attempt2.id

        with SessionLocal() as db:
            remediate_chapter(chapter_id, user_id, ["recursion-base-case"], attempt2_id, db)

    with SessionLocal() as db:
        versions = sorted(
            c.version for c in db.query(ChapterContent).filter_by(
                chapter_id=chapter_id, scope="user", user_id=user_id,
            ).all()
        )
        assert versions == [2, 3]


def test_redelivery_for_same_attempt_does_not_duplicate_version():
    with SessionLocal() as db:
        chapter_id, attempt_id, user_id = _make_chapter_with_v1_and_attempt(db, "remediate-c")

    with patch("app.agents.chapter_content.remediate.generate_remediation_outline", return_value=_OUTLINE) as mock_outline, \
         patch("app.agents.chapter_content.remediate.generate_chapter_section", return_value=_SECTION_RESPONSE), \
         patch("app.agents.chapter_content.remediate.generate_chapter_assignment_task") as mock_assignment_task:
        with SessionLocal() as db:
            remediate_chapter(chapter_id, user_id, ["recursion-base-case"], attempt_id, db)
        with SessionLocal() as db:
            remediate_chapter(chapter_id, user_id, ["recursion-base-case"], attempt_id, db)  # redelivery

    mock_outline.assert_called_once()  # second call adopted the existing "ready" row and returned immediately
    mock_assignment_task.delay.assert_called_once()
    with SessionLocal() as db:
        count = db.query(ChapterContent).filter_by(
            chapter_id=chapter_id, scope="user", user_id=user_id, remediation_source_attempt_id=attempt_id,
        ).count()
        assert count == 1


def test_outline_failure_marks_content_failed():
    with SessionLocal() as db:
        chapter_id, attempt_id, user_id = _make_chapter_with_v1_and_attempt(db, "remediate-d")

    with patch("app.agents.chapter_content.remediate.generate_remediation_outline", side_effect=RuntimeError("llm down")), \
         patch("app.agents.chapter_content.remediate.generate_chapter_assignment_task") as mock_assignment_task:
        with SessionLocal() as db:
            remediate_chapter(chapter_id, user_id, ["recursion-base-case"], attempt_id, db)

    with SessionLocal() as db:
        content = db.query(ChapterContent).filter_by(
            chapter_id=chapter_id, scope="user", user_id=user_id,
        ).one()
        assert content.status == "failed"
        assert content.error == "llm down"
    mock_assignment_task.delay.assert_not_called()


def test_remediation_runs_research_with_weak_concepts_and_passes_notes():
    with SessionLocal() as db:
        chapter_id, attempt_id, user_id = _make_chapter_with_v1_and_attempt(db, "remediate-research")

    with patch("app.agents.chapter_content.remediate.generate_remediation_outline", return_value=_OUTLINE), \
         patch("app.agents.chapter_content.remediate.generate_chapter_section", return_value=_SECTION_RESPONSE) as mock_section, \
         patch("app.agents.chapter_content.remediate.generate_chapter_assignment_task"), \
         patch("app.agents.chapter_content.research.get_chat_model"), \
         patch("app.agents.chapter_content.research.run_web_research", return_value="NOTES") as mock_research:
        with SessionLocal() as db:
            remediate_chapter(chapter_id, user_id, ["recursion-base-case"], attempt_id, db)

    mock_research.assert_called_once()
    assert mock_section.call_args.args[5] == "NOTES"
    with SessionLocal() as db:
        content = db.query(ChapterContent).filter_by(
            chapter_id=chapter_id, scope="user", user_id=user_id,
        ).one()
        assert content.research_notes == "NOTES"
