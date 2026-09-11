from datetime import datetime, timezone
from app.db import SessionLocal
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Course, Module, Chapter
from app.models.enrollment import UserCourse
from app.models.user import User
from app.services.export_tools import assignments as assignment_tools
from app.services.export_tools import content as content_tools
from app.services.export_tools._errors import ExportToolError
from app.services.export_tools.registry import build_export_tools
import pytest


def _seed_tool_course():
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        db.add(User(id=101, email="export-tools-one@example.com", password_hash="x"))
        db.add(User(id=102, email="export-tools-two@example.com", password_hash="x"))
        course = Course(topic_slug="export-tools", topic_raw="Tools",
                        topic_embedding=[0.0] * 2048, created_at=now)
        db.add(course)
        db.commit()
        db.add(UserCourse(user_id=101, course_id=course.id, status="in_progress", progress=0.0,
                          enrolled_at=now, last_opened_at=now))
        module = Module(course_id=course.id, title="Global", objective="o", order=1, scope="global")
        db.add(module)
        db.commit()
        chapter = Chapter(module_id=module.id, title="C", objective="o", order=1, scope="global")
        db.add(chapter)
        db.commit()
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                 outline=[], created_at=now, updated_at=now)
        db.add(content)
        db.commit()
        db.refresh(content)
        db.add(ChapterContentSection(chapter_content_id=content.id, order=0, heading="H",
                                    kind="teaching", body_markdown="Body",
                                    examples=[{"prompt": "P", "walkthrough": "W"}]))
        db.commit()
        assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                                status="ready", created_at=now, updated_at=now)
        db.add(assignment)
        db.commit()
        db.refresh(assignment)
        db.add(AssignmentQuestion(assignment_id=assignment.id, order=1, type="mcq", text="Q",
                                  options=["A", "B"], correct_answer="A", explanation="Because A.",
                                  concept_tag="concept", difficulty="easy"))
        attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=101, status="graded",
                                    overall_score=0.4, created_at=now, updated_at=now)
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
        remediation = ChapterContent(chapter_id=chapter.id, version=2, scope="user", user_id=101,
                                     status="ready", outline=[], remediation_target_tags=["concept"],
                                     remediation_source_attempt_id=attempt.id,
                                     created_at=now, updated_at=now)
        db.add(remediation)
        db.commit()
        db.refresh(remediation)
        db.add(ChapterContentSection(chapter_content_id=remediation.id, order=0, heading="Remediation",
                                    kind="teaching", body_markdown="Body",
                                    examples=[{"prompt": "P", "walkthrough": "W"}]))
        db.commit()
        bucket = Module(course_id=course.id, title="Additional Chapters", objective="o", order=2,
                        scope="user", user_id=102)
        db.add(bucket)
        db.commit()
        hidden = Chapter(module_id=bucket.id, title="Hidden", objective="o", order=1,
                         scope="user", user_id=102)
        db.add(hidden)
        db.commit()
        return course.topic_slug, chapter.id, assignment.id


def test_tools_expose_versions_without_leaking_another_learner_bucket():
    slug, chapter_id, _ = _seed_tool_course()
    with SessionLocal() as db:
        versions = content_tools.get_chapter_versions(db, 101, slug, chapter_id)
        assert versions["version_count"] == 2
        assert [v["version"] for v in versions["versions"]] == [1, 2]
        everyone = content_tools.list_all_chapters(db, 101, slug)
        assert [c["title"] for c in everyone] == ["C"]


def test_specific_and_batch_content_have_expected_shape():
    slug, chapter_id, _ = _seed_tool_course()
    with SessionLocal() as db:
        specific = content_tools.get_chapter_version_content(db, 101, slug, chapter_id, version=2)
        assert specific["available"] is True
        assert specific["version"] == 2
        assert specific["sections"][0]["body_markdown"] == "Body"
        batch = content_tools.get_chapters_content(db, 101, slug, [chapter_id])
        assert batch[0]["chapter_id"] == chapter_id
        assert batch[0]["content"]["available"] is True


def test_assignment_answers_stay_inside_course_scoped_tool():
    slug, _, assignment_id = _seed_tool_course()
    with SessionLocal() as db:
        questions = assignment_tools.get_assignment_questions(db, 101, slug, assignment_id)
        assert questions["questions"][0]["correct_answer"] == "A"
        with pytest.raises(ExportToolError):
            assignment_tools.get_assignment_questions(db, 102, slug, assignment_id)


def test_assignment_questions_exclude_other_learners_topups():
    slug, chapter_id, _ = _seed_tool_course()
    with SessionLocal() as db:
        chapter = db.get(Chapter, chapter_id)
        assert chapter is not None
        now = datetime.now(timezone.utc)
        module_assignment = Assignment(level="module", module_id=chapter.module_id, scope="global",
                                       status="ready", created_at=now, updated_at=now)
        db.add(module_assignment)
        db.commit()
        db.refresh(module_assignment)
        db.add(AssignmentQuestion(assignment_id=module_assignment.id, order=1, type="mcq",
                                  text="Global", options=["A", "B"], correct_answer="A",
                                  explanation="g", concept_tag="concept", difficulty="easy",
                                  user_id=None))
        db.add(AssignmentQuestion(assignment_id=module_assignment.id, order=2, type="mcq",
                                  text="Other learner", options=["A", "B"], correct_answer="B",
                                  explanation="leak", concept_tag="concept", difficulty="easy",
                                  user_id=102))
        db.commit()
        questions = assignment_tools.get_assignment_questions(db, 101, slug, module_assignment.id)
        assert [q["text"] for q in questions["questions"]] == ["Global"]


def test_listings_exclude_other_learners_user_scoped_chapters():
    slug, chapter_id, _ = _seed_tool_course()
    with SessionLocal() as db:
        chapter = db.get(Chapter, chapter_id)
        assert chapter is not None
        db.add(Chapter(module_id=chapter.module_id, title="Other learner", objective="o", order=2,
                       scope="user", user_id=102))
        db.commit()
        assert [c["title"] for c in content_tools.list_all_chapters(db, 101, slug)] == ["C"]
        listed = content_tools.list_chapters(db, 101, slug, chapter.module_id)
        assert [c["title"] for c in listed] == ["C"]


def test_registry_contains_all_eight_course_scoped_tools():
    with SessionLocal() as db:
        names = sorted(t.name for t in build_export_tools(db, 101, "export-tools"))
    assert names == sorted([
        "list_modules",
        "list_chapters",
        "list_all_chapters",
        "get_chapter_versions",
        "get_chapter_version_content",
        "get_chapters_content",
        "list_assignments",
        "get_assignment_questions",
    ])
