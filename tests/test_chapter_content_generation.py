from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.user import User
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.agents.chapter_content.generate import stream_chapter_content
from app.agents.chapter_content.nodes.generate_section_outline import SectionOutlineDraft
from app.agents.chapter_content.nodes.generate_chapter_section import ChapterSectionResponse, ExampleDraft


def _make_chapter(topic_slug: str) -> Chapter:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=topic_slug, topic_raw=topic_slug, topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="Generic Functions", objective="Write reusable code", order=1)
        db.add(chapter)
        db.commit()
        db.refresh(chapter)
        return chapter


def test_generates_all_sections_and_dispatches_diagram_task():
    chapter = _make_chapter("stream-cc-full")
    outline = [
        SectionOutlineDraft(heading="Intro", objective="o1", kind="intro", order=1),
        SectionOutlineDraft(heading="Deep Dive", objective="o2", kind="teaching", order=2),
    ]
    section_responses = [
        ChapterSectionResponse(body_markdown="intro body", examples=[], diagram_spec=None),
        ChapterSectionResponse(
            body_markdown="deep dive body",
            examples=[ExampleDraft(prompt="p", walkthrough="w")],
            diagram_spec={"nodes": [{"id": "a", "label": "A"}], "edges": []},  # type: ignore[arg-type]
        ),
    ]

    with SessionLocal() as db, \
         patch("app.agents.chapter_content.generate.generate_section_outline", return_value=outline), \
         patch("app.agents.chapter_content.generate.generate_chapter_section", side_effect=section_responses), \
         patch("app.agents.chapter_content.generate.render_diagram_task") as mock_diagram_task, \
         patch("app.agents.chapter_content.generate.generate_chapter_assignment_task") as mock_assignment_task:
        events = list(stream_chapter_content(chapter, db, user_id=1))

    section_events = [e for e in events if e["type"] == "section_ready"]
    assert len(section_events) == 2
    assert section_events[0]["order"] == 0
    assert section_events[0]["diagram_status"] is None
    assert section_events[1]["order"] == 1
    assert section_events[1]["diagram_status"] == "pending"
    assert events[-1]["type"] == "done"
    mock_diagram_task.delay.assert_called_once()
    mock_assignment_task.delay.assert_called_once()


def test_ready_content_replays_without_any_llm_calls():
    chapter = _make_chapter("stream-cc-replay")
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                  outline=[{"heading": "H", "objective": "o", "kind": "teaching"}],
                                  created_at=now, updated_at=now)
        db.add(content)
        db.commit()
        db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H", kind="teaching",
                                      body_markdown="cached body",
                                      examples=[{"prompt": "p", "walkthrough": "w"}]))
        db.commit()

    with SessionLocal() as db, \
         patch("app.agents.chapter_content.generate.generate_section_outline") as mock_outline, \
         patch("app.agents.chapter_content.generate.generate_chapter_section") as mock_section:
        events = list(stream_chapter_content(chapter, db, user_id=1))

    mock_outline.assert_not_called()
    mock_section.assert_not_called()
    assert [e["type"] for e in events] == ["section_ready", "done"]
    assert events[0]["body_markdown"] == "cached body"


def test_resumes_reuses_persisted_outline_and_skips_done_sections():
    chapter = _make_chapter("stream-cc-resume")
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        content = ChapterContent(
            chapter_id=chapter.id, version=1, scope="global", status="failed",
            outline=[
                {"heading": "H1", "objective": "o1", "kind": "teaching"},
                {"heading": "H2", "objective": "o2", "kind": "teaching"},
            ],
            error="prior transient failure", created_at=now, updated_at=now,
        )
        db.add(content)
        db.commit()
        db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H1", kind="teaching",
                                      body_markdown="already done",
                                      examples=[{"prompt": "p", "walkthrough": "w"}]))
        db.commit()

    second_section = ChapterSectionResponse(
        body_markdown="second body", examples=[ExampleDraft(prompt="p2", walkthrough="w2")], diagram_spec=None,
    )
    with SessionLocal() as db, \
         patch("app.agents.chapter_content.generate.generate_section_outline") as mock_outline, \
         patch("app.agents.chapter_content.generate.generate_chapter_section", return_value=second_section) as mock_section, \
         patch("app.agents.chapter_content.generate.generate_chapter_assignment_task") as mock_assignment_task:
        events = list(stream_chapter_content(chapter, db, user_id=1))

    mock_outline.assert_not_called()  # outline was already persisted, must not be regenerated
    mock_section.assert_called_once()  # only the missing section (H2) is generated
    mock_assignment_task.delay.assert_called_once()
    section_events = [e for e in events if e["type"] == "section_ready"]
    assert section_events[0]["body_markdown"] == "already done"
    assert section_events[1]["body_markdown"] == "second body"
    assert events[-1]["type"] == "done"


def test_generation_failure_marks_content_failed_and_yields_error():
    chapter = _make_chapter("stream-cc-fail")
    outline = [SectionOutlineDraft(heading="H", objective="o", kind="teaching", order=1)]

    with SessionLocal() as db, \
         patch("app.agents.chapter_content.generate.generate_section_outline", return_value=outline), \
         patch("app.agents.chapter_content.generate.generate_chapter_section", side_effect=RuntimeError("llm down")):
        events = list(stream_chapter_content(chapter, db, user_id=1))

    assert events[-1]["type"] == "error"
    with SessionLocal() as db:
        content = db.query(ChapterContent).filter_by(chapter_id=chapter.id).one()
        assert content.status == "failed"
        assert content.error == "llm down"


def _make_extension_chapter(topic_slug: str) -> Chapter:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=topic_slug, topic_raw=topic_slug, topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.flush()
        user = User(email=f"{topic_slug}@example.com", password_hash="x")
        db.add(user)
        db.flush()
        module = Module(course_id=course.id, title="Additional Chapters", objective="o", order=2,
                        scope="user", user_id=user.id)
        db.add(module)
        db.flush()
        chapter = Chapter(module_id=module.id, title="Ext Functions", objective="Extend the course",
                          order=1, scope="user", user_id=user.id)
        db.add(chapter)
        db.commit()
        db.refresh(chapter)
        return chapter


def test_extension_chapter_generates_user_scoped_base_content():
    chapter = _make_extension_chapter("stream-cc-extension")
    outline = [SectionOutlineDraft(heading="Intro", objective="o1", kind="intro", order=1)]
    section_response = ChapterSectionResponse(body_markdown="ext body", examples=[], diagram_spec=None)

    assert chapter.user_id is not None
    with SessionLocal() as db, \
         patch("app.agents.chapter_content.generate.generate_section_outline", return_value=outline) as mock_outline, \
         patch("app.agents.chapter_content.generate.generate_chapter_section", return_value=section_response) as mock_section, \
         patch("app.agents.chapter_content.generate.generate_chapter_assignment_task"):
        events = list(stream_chapter_content(chapter, db, user_id=chapter.user_id))

    mock_outline.assert_called_once()
    mock_section.assert_called_once()
    section_events = [e for e in events if e["type"] == "section_ready"]
    assert [e["body_markdown"] for e in section_events] == ["ext body"]
    assert events[-1]["type"] == "done"

    with SessionLocal() as db:
        content = db.query(ChapterContent).filter_by(chapter_id=chapter.id).one()
        assert content.scope == "user"
        assert content.user_id == chapter.user_id
        assert content.status == "ready"
        sections = db.query(ChapterContentSection).filter_by(chapter_content_id=content.id).all()
        assert len(sections) == 1
        assert sections[0].body_markdown == "ext body"


def test_remediation_content_is_not_generated_inline():
    chapter = _make_chapter("stream-cc-remediation-gate")
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        user = User(email="remediation-gate@example.com", password_hash="x")
        db.add(user)
        db.flush()
        base = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                              outline=[], created_at=now, updated_at=now)
        db.add(base)
        db.flush()
        assignment = Assignment(level="chapter", chapter_content_id=base.id, scope="global",
                                status="ready", created_at=now, updated_at=now)
        db.add(assignment)
        db.flush()
        attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                    overall_score=0.5, created_at=now, updated_at=now)
        db.add(attempt)
        db.flush()
        remediation = ChapterContent(chapter_id=chapter.id, version=2, scope="user", user_id=user.id,
                                      status="generating", outline=[], remediation_target_tags=["t"],
                                      remediation_source_attempt_id=attempt.id,
                                      created_at=now, updated_at=now)
        db.add(remediation)
        db.commit()
        user_id = user.id

    with SessionLocal() as db, \
         patch("app.agents.chapter_content.generate.generate_section_outline") as mock_outline, \
         patch("app.agents.chapter_content.generate.generate_chapter_section") as mock_section:
        events = list(stream_chapter_content(chapter, db, user_id=user_id))

    mock_outline.assert_not_called()
    mock_section.assert_not_called()
    assert {"type": "generating"} in events
    assert events[-1]["type"] == "done"


def test_replaying_already_ready_content_does_not_redispatch_assignment():
    chapter = _make_chapter("stream-cc-replay-no-redispatch")
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                  outline=[{"heading": "H", "objective": "o", "kind": "teaching"}],
                                  created_at=now, updated_at=now)
        db.add(content)
        db.commit()
        db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H", kind="teaching",
                                      body_markdown="cached body",
                                      examples=[{"prompt": "p", "walkthrough": "w"}]))
        db.commit()

    with SessionLocal() as db, \
         patch("app.agents.chapter_content.generate.generate_chapter_assignment_task") as mock_assignment_task:
        list(stream_chapter_content(chapter, db, user_id=1))

    mock_assignment_task.delay.assert_not_called()
