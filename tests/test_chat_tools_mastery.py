from datetime import datetime, timezone

from app.db import SessionLocal
from app.models.chapter_content import ChapterContent
from app.models.course import Chapter, Concept, Course, Module
from app.models.enrollment import UserCourse
from app.models.user import User
from app.services.chat_tools import mastery as mastery_tools


def _now():
    return datetime.now(timezone.utc)


def _make_course_with_two_chapters(db, slug: str, user_id: int):
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
    db.add(course); db.flush()
    module = Module(course_id=course.id, title="M1", objective="o", order=1)
    db.add(module); db.flush()
    chapter_a = Chapter(module_id=module.id, title="C1", objective="o", order=1)
    chapter_b = Chapter(module_id=module.id, title="C2", objective="o", order=2)
    db.add_all([chapter_a, chapter_b]); db.flush()
    db.add(Concept(course_id=course.id, name="concept-a", chapter_id=chapter_a.id))
    db.add(Concept(course_id=course.id, name="concept-b", chapter_id=chapter_b.id))
    db.add(UserCourse(user_id=user_id, course_id=course.id, enrolled_at=_now(), last_opened_at=_now()))
    db.flush()
    return course, module, chapter_a, chapter_b


def test_get_weak_concepts_by_chapter_scopes_to_one_chapter():
    with SessionLocal() as db:
        user = User(email="mt-a@example.com", password_hash="x")
        db.add(user); db.flush()
        course, module, chapter_a, chapter_b = _make_course_with_two_chapters(db, "mt-course-a", user.id)
        db.add(ChapterContent(chapter_id=chapter_a.id, version=1, scope="global", status="ready",
                               outline=[], created_at=_now(), updated_at=_now()))
        db.add(ChapterContent(chapter_id=chapter_b.id, version=1, scope="global", status="ready",
                               outline=[], created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, slug, chapter_a_id = user.id, course.topic_slug, chapter_a.id

    with SessionLocal() as db:
        result = mastery_tools.get_weak_concepts_by_chapter(db, user_id, slug, chapter_a_id)
    assert result == {"concept-a": "strong"}


def test_get_recurring_weak_concepts_counts_across_versions():
    with SessionLocal() as db:
        user = User(email="mt-b@example.com", password_hash="x")
        db.add(user); db.flush()
        course, module, chapter_a, chapter_b = _make_course_with_two_chapters(db, "mt-course-b", user.id)
        for version, tags in [(1, ["concept-a"]), (2, ["concept-a"]), (3, ["concept-a"])]:
            db.add(ChapterContent(
                chapter_id=chapter_a.id, version=version, scope="global", status="ready", outline=[],
                remediation_target_tags=tags if version > 1 else None,
                created_at=_now(), updated_at=_now(),
            ))
        db.commit()
        user_id, slug = user.id, course.topic_slug

    with SessionLocal() as db:
        result = mastery_tools.get_recurring_weak_concepts(db, user_id, slug, min_occurrences=2)
    assert result == [{"concept_tag": "concept-a", "occurrences": 2}]
