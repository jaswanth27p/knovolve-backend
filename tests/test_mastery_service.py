from datetime import datetime, timezone
from app.db import SessionLocal
from app.models.user import User
from app.models.course import Course, Module, Chapter, Concept
from app.models.chapter_content import ChapterContent
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.services import mastery


def _now():
    return datetime.now(timezone.utc)


def _make_course_with_chapters(db, slug: str, n_chapters: int):
    """Mirrors test_progression_service.py's helper: each chapter gets a
    global V1 ChapterContent + its global chapter assignment."""
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
    db.add(course); db.flush()
    module = Module(course_id=course.id, title="M", objective="o", order=1)
    db.add(module); db.flush()
    chapters = []
    for i in range(n_chapters):
        chapter = Chapter(module_id=module.id, title=f"C{i}", objective="o", order=i)
        db.add(chapter); db.flush()
        content = ChapterContent(chapter_id=chapter.id, version=1, scope="global", status="ready",
                                  outline=[], created_at=_now(), updated_at=_now())
        db.add(content); db.flush()
        assignment = Assignment(level="chapter", chapter_content_id=content.id, scope="global",
                                 status="ready", created_at=_now(), updated_at=_now())
        db.add(assignment); db.flush()
        chapters.append((chapter, content, assignment))
    return course, module, chapters


def test_get_weak_concept_tags_unpassed_chapter_contributes_passed_chapter_does_not():
    with SessionLocal() as db:
        course, module, chapters = _make_course_with_chapters(db, "mastery-a", 2)
        chapter_a, content_a, assignment_a = chapters[0]
        chapter_b, content_b, assignment_b = chapters[1]

        content_a.remediation_target_tags = ["tag-a1", "tag-a2"]
        content_b.remediation_target_tags = ["tag-b1"]
        db.flush()

        user = User(email="mastery-a@example.com", password_hash="x")
        db.add(user); db.flush()

        # Chapter A: no graded attempt at all -> not passed.
        # Chapter B: a passing graded attempt -> passed, contributes nothing.
        db.add(AssignmentAttempt(assignment_id=assignment_b.id, user_id=user.id, status="graded",
                                  overall_score=0.9, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, course_id = user.id, course.id

    with SessionLocal() as db:
        tags = mastery.get_weak_concept_tags(db, user_id, course_id)
        assert tags == {"tag-a1", "tag-a2"}


def test_never_attempted_chapter_contributes_nothing_and_is_unassessed():
    with SessionLocal() as db:
        course = Course(topic_slug="mastery-b", topic_raw="mastery-b", topic_embedding=[0.0] * 2048,
                         created_at=_now())
        db.add(course); db.flush()
        module = Module(course_id=course.id, title="M", objective="o", order=1)
        db.add(module); db.flush()
        chapter = Chapter(module_id=module.id, title="C0", objective="o", order=0)
        db.add(chapter); db.flush()
        # No ChapterContent row at all for this chapter -> never attempted.
        db.add(Concept(course_id=course.id, name="concept-x", chapter_id=chapter.id))
        db.add(Concept(course_id=course.id, name="concept-y", chapter_id=chapter.id))
        db.flush()

        user = User(email="mastery-b@example.com", password_hash="x")
        db.add(user); db.flush()
        db.commit()
        user_id, course_id = user.id, course.id

    with SessionLocal() as db:
        tags = mastery.get_weak_concept_tags(db, user_id, course_id)
        assert tags == set()

        statuses = mastery.get_concept_statuses(db, user_id, course_id)
        assert statuses == {"concept-x": "unassessed", "concept-y": "unassessed"}


def test_get_concept_statuses_returns_all_three_bands():
    with SessionLocal() as db:
        course, module, chapters = _make_course_with_chapters(db, "mastery-c", 2)
        chapter_weak, content_weak, assignment_weak = chapters[0]
        chapter_strong, content_strong, assignment_strong = chapters[1]

        content_weak.remediation_target_tags = ["weak-concept"]
        db.flush()

        db.add(Concept(course_id=course.id, name="weak-concept", chapter_id=chapter_weak.id))
        db.add(Concept(course_id=course.id, name="strong-concept", chapter_id=chapter_strong.id))
        # A third, never-attempted chapter for the "unassessed" band.
        chapter_unassessed = Chapter(module_id=module.id, title="C-unassessed", objective="o", order=99)
        db.add(chapter_unassessed); db.flush()
        db.add(Concept(course_id=course.id, name="unassessed-concept", chapter_id=chapter_unassessed.id))
        db.flush()

        user = User(email="mastery-c@example.com", password_hash="x")
        db.add(user); db.flush()

        # weak chapter: graded but failing attempt -> not passed, tags stay weak.
        db.add(AssignmentAttempt(assignment_id=assignment_weak.id, user_id=user.id, status="graded",
                                  overall_score=0.2, created_at=_now(), updated_at=_now()))
        # strong chapter: graded passing attempt -> passed.
        db.add(AssignmentAttempt(assignment_id=assignment_strong.id, user_id=user.id, status="graded",
                                  overall_score=0.85, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, course_id = user.id, course.id

    with SessionLocal() as db:
        statuses = mastery.get_concept_statuses(db, user_id, course_id)
        assert statuses == {
            "weak-concept": "weak",
            "strong-concept": "strong",
            "unassessed-concept": "unassessed",
        }


def test_get_module_lagged_history_includes_only_versions_with_tags_regardless_of_pass_status():
    with SessionLocal() as db:
        course, module, chapters = _make_course_with_chapters(db, "mastery-d", 1)
        chapter, v1_content, v1_assignment = chapters[0]
        # V1 has no remediation tags (nothing triggered a remediation yet).
        assert v1_content.remediation_target_tags is None

        user = User(email="mastery-d@example.com", password_hash="x")
        db.add(user); db.flush()

        # The V1 attempt that triggered the V2 remediation version.
        triggering_attempt = AssignmentAttempt(assignment_id=v1_assignment.id, user_id=user.id, status="graded",
                                                overall_score=0.3, created_at=_now(), updated_at=_now())
        db.add(triggering_attempt); db.flush()

        v2_content = ChapterContent(chapter_id=chapter.id, version=2, scope="user", user_id=user.id,
                                     status="ready", outline=[], remediation_target_tags=["lagged-concept"],
                                     remediation_source_attempt_id=triggering_attempt.id,
                                     created_at=_now(), updated_at=_now())
        db.add(v2_content); db.flush()
        v2_assignment = Assignment(level="chapter", chapter_content_id=v2_content.id, scope="user",
                                    user_id=user.id, status="ready", created_at=_now(), updated_at=_now())
        db.add(v2_assignment); db.flush()

        # V2 attempt passes -> chapter_passed becomes True, so this chapter
        # would contribute nothing to get_weak_concept_tags any more, but
        # the lagged-history query ignores pass status entirely.
        db.add(AssignmentAttempt(assignment_id=v2_assignment.id, user_id=user.id, status="graded",
                                  overall_score=0.95, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, module_id, course_id, triggering_attempt_id = user.id, module.id, course.id, triggering_attempt.id
        chapter_id = chapter.id

    with SessionLocal() as db:
        from app.services import progression
        # Chapter is passed now, so it stopped contributing to the
        # "currently weak" view entirely...
        assert progression._chapter_passed(db, chapter_id, user_id) is True
        assert mastery.get_weak_concept_tags(db, user_id, course_id) == set()

    with SessionLocal() as db:
        # ...but the historical query ignores pass status by design, and
        # still shows the concepts that were lagged along the way.
        history = mastery.get_module_lagged_history(db, user_id, module_id)
        assert len(history) == 1
        entry = history[0]
        assert entry["chapter_id"] == chapter_id
        assert entry["version"] == 2
        assert entry["concept_tags"] == ["lagged-concept"]
        assert entry["triggered_by_attempt_id"] == triggering_attempt_id
