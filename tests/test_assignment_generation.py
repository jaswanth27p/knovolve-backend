# backend/tests/test_assignment_generation.py
from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.assignment import Assignment, AssignmentQuestion
from app.agents.assignment.nodes.generate_section_questions import QuestionDraft
from app.agents.assignment.generate import generate_chapter_assignment
from app.tasks.assignment_tasks import generate_chapter_assignment_task


def _make_ready_chapter_content(slug: str, section_count: int = 1) -> int:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="Chapter T", objective="Chapter O", order=1)
        db.add(chapter)
        db.commit()
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                  outline=[], created_at=now, updated_at=now)
        db.add(content)
        db.commit()
        for i in range(section_count):
            db.add(ChapterContentSection(
                chapter_content_id=content.id, order=i, heading=f"Section {i}", kind="teaching",
                body_markdown=f"body {i}", examples=[{"prompt": "p", "walkthrough": "w"}],
            ))
        db.commit()
        return content.id


def _draft(tag: str, qtype: str = "mcq") -> QuestionDraft:
    options = ["a", "b"] if qtype == "mcq" else None
    return QuestionDraft(type=qtype, text=f"q-{tag}", options=options, correct_answer="a",
                        explanation="e", concept_tag=tag, difficulty="easy")


def test_generates_questions_per_section_no_topup_needed():
    content_id = _make_ready_chapter_content("asg-gen-a", section_count=2)
    section_drafts = [[_draft("s0-c1"), _draft("s0-c2")], [_draft("s1-c1")]]

    with patch("app.agents.assignment.generate.generate_questions_for_section", side_effect=section_drafts), \
         patch("app.agents.assignment.generate.generate_topup_questions") as mock_topup:
        with SessionLocal() as db:
            generate_chapter_assignment(content_id, db)

    mock_topup.assert_not_called()
    with SessionLocal() as db:
        assignment = db.query(Assignment).filter_by(chapter_content_id=content_id).one()
        assert assignment.status == "ready"
        questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment.id).order_by(AssignmentQuestion.order).all()
        assert len(questions) == 3
        assert questions[0].concept_tag == "s0-c1"
        assert questions[0].source_section_id is not None


def test_tops_up_when_below_minimum():
    content_id = _make_ready_chapter_content("asg-gen-b", section_count=1)

    with patch("app.agents.assignment.generate.generate_questions_for_section", return_value=[_draft("only")]), \
         patch("app.agents.assignment.generate.generate_topup_questions",
               return_value=[_draft("topup-1"), _draft("topup-2")]) as mock_topup:
        with SessionLocal() as db:
            generate_chapter_assignment(content_id, db)

    mock_topup.assert_called_once()
    assert mock_topup.call_args.args[-1] == 2  # needed = 3 - 1
    with SessionLocal() as db:
        assignment = db.query(Assignment).filter_by(chapter_content_id=content_id).one()
        questions = db.query(AssignmentQuestion).filter_by(assignment_id=assignment.id).all()
        assert len(questions) == 3
        assert any(q.source_section_id is None for q in questions)  # topup questions


def test_idempotent_when_already_ready():
    content_id = _make_ready_chapter_content("asg-gen-c", section_count=1)
    with patch("app.agents.assignment.generate.generate_questions_for_section", return_value=[_draft("a"), _draft("b"), _draft("c")]):
        with SessionLocal() as db:
            generate_chapter_assignment(content_id, db)

    with patch("app.agents.assignment.generate.generate_questions_for_section") as mock_gen:
        with SessionLocal() as db:
            generate_chapter_assignment(content_id, db)  # second call, already ready

    mock_gen.assert_not_called()


def test_failure_marks_assignment_failed_with_no_partial_questions():
    content_id = _make_ready_chapter_content("asg-gen-d", section_count=1)
    with patch("app.agents.assignment.generate.generate_questions_for_section", side_effect=RuntimeError("llm down")):
        with SessionLocal() as db:
            generate_chapter_assignment(content_id, db)

    with SessionLocal() as db:
        assignment = db.query(Assignment).filter_by(chapter_content_id=content_id).one()
        assert assignment.status == "failed"
        assert assignment.error == "llm down"
        assert db.query(AssignmentQuestion).filter_by(assignment_id=assignment.id).count() == 0


def test_failed_assignment_regenerates_from_scratch_on_retry():
    content_id = _make_ready_chapter_content("asg-gen-e", section_count=1)
    with patch("app.agents.assignment.generate.generate_questions_for_section", side_effect=RuntimeError("llm down")):
        with SessionLocal() as db:
            generate_chapter_assignment(content_id, db)

    with patch("app.agents.assignment.generate.generate_questions_for_section",
               return_value=[_draft("a"), _draft("b"), _draft("c")]):
        with SessionLocal() as db:
            generate_chapter_assignment(content_id, db)

    with SessionLocal() as db:
        assignment = db.query(Assignment).filter_by(chapter_content_id=content_id).one()
        assert assignment.status == "ready"
        assert assignment.error is None
        assert db.query(AssignmentQuestion).filter_by(assignment_id=assignment.id).count() == 3


def test_task_wrapper_invokes_generation():
    content_id = _make_ready_chapter_content("asg-gen-f", section_count=1)
    with patch("app.tasks.assignment_tasks.generate_chapter_assignment") as mock_generate:
        generate_chapter_assignment_task(content_id)  # pyright: ignore[reportCallIssue]

    mock_generate.assert_called_once()
    assert mock_generate.call_args.args[0] == content_id
