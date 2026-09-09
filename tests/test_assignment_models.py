from datetime import datetime, timezone
import pytest
from sqlalchemy.exc import IntegrityError
from app.db import SessionLocal
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.models.user import User
from app.models.assignment import Assignment, AssignmentQuestion, AssignmentUserTopup


def _now():
    return datetime.now(timezone.utc)


def _make_chapter_content(db, slug: str, scope: str = "global", user_id: int | None = None) -> tuple[int, int]:
    course = Course(topic_slug=slug, topic_raw=slug, topic_embedding=[0.0] * 2048, created_at=_now())
    db.add(course)
    db.flush()
    module = Module(course_id=course.id, title="M", objective="o", order=1)
    db.add(module)
    db.flush()
    chapter = Chapter(module_id=module.id, title="C", objective="o", order=1)
    db.add(chapter)
    db.flush()
    content = ChapterContent(chapter_id=chapter.id, version=1, scope=scope, user_id=user_id,
                              status="ready", outline=[], created_at=_now(), updated_at=_now())
    db.add(content)
    db.flush()
    return content.id, module.id


def test_chapter_level_check_constraint_rejects_module_id_set():
    with SessionLocal() as db:
        content_id, _ = _make_chapter_content(db, "asg-model-a")
        db.commit()
    with SessionLocal() as db:
        db.add(Assignment(level="chapter", chapter_content_id=content_id, module_id=1,
                          scope="global", status="generating", created_at=_now(), updated_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_one_global_assignment_per_chapter_content():
    with SessionLocal() as db:
        content_id, _ = _make_chapter_content(db, "asg-model-b")
        db.commit()
    with SessionLocal() as db:
        db.add(Assignment(level="chapter", chapter_content_id=content_id, scope="global",
                          status="generating", created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(Assignment(level="chapter", chapter_content_id=content_id, scope="global",
                          status="generating", created_at=_now(), updated_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_global_and_user_assignment_can_coexist_for_different_chapter_content():
    with SessionLocal() as db:
        user = User(email="asg-model-c@example.com", password_hash="x")
        db.add(user)
        db.flush()
        global_content_id, _ = _make_chapter_content(db, "asg-model-c")
        user_content_id, _ = _make_chapter_content(db, "asg-model-c-v2", scope="user", user_id=user.id)
        db.add(Assignment(level="chapter", chapter_content_id=global_content_id, scope="global",
                          status="generating", created_at=_now(), updated_at=_now()))
        db.add(Assignment(level="chapter", chapter_content_id=user_content_id, scope="user",
                          user_id=user.id, status="generating", created_at=_now(), updated_at=_now()))
        db.commit()  # must not raise


def test_one_global_assignment_per_module():
    with SessionLocal() as db:
        _, module_id = _make_chapter_content(db, "asg-model-d")
        db.commit()
    with SessionLocal() as db:
        db.add(Assignment(level="module", module_id=module_id, scope="global",
                          status="generating", created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(Assignment(level="module", module_id=module_id, scope="global",
                          status="generating", created_at=_now(), updated_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()


def test_assignment_question_unique_order_per_assignment():
    with SessionLocal() as db:
        content_id, _ = _make_chapter_content(db, "asg-model-e")
        assignment = Assignment(level="chapter", chapter_content_id=content_id, scope="global",
                                status="ready", created_at=_now(), updated_at=_now())
        db.add(assignment)
        db.commit()
        db.add(AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q1",
                                  options=["a", "b"], correct_answer="a", explanation="e",
                                  concept_tag="t", difficulty="easy"))
        db.commit()
        db.add(AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q2",
                                  options=["a", "b"], correct_answer="a", explanation="e",
                                  concept_tag="t", difficulty="easy"))
        with pytest.raises(IntegrityError):
            db.commit()


def test_assignment_question_user_id_null_round_trips_as_before():
    with SessionLocal() as db:
        content_id, _ = _make_chapter_content(db, "asg-model-f")
        assignment = Assignment(level="chapter", chapter_content_id=content_id, scope="global",
                                status="ready", created_at=_now(), updated_at=_now())
        db.add(assignment)
        db.commit()
        question = AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q1",
                                      options=["a", "b"], correct_answer="a", explanation="e",
                                      concept_tag="t", difficulty="easy")
        db.add(question)
        db.commit()
        question_id = question.id

    with SessionLocal() as db:
        fetched = db.get(AssignmentQuestion, question_id)
        assert fetched is not None
        assert fetched.user_id is None


def test_assignment_question_user_id_set_persists_and_reads_back():
    with SessionLocal() as db:
        user = User(email="asg-model-f2@example.com", password_hash="x")
        db.add(user)
        db.flush()
        content_id, _ = _make_chapter_content(db, "asg-model-f2")
        assignment = Assignment(level="chapter", chapter_content_id=content_id, scope="global",
                                status="ready", created_at=_now(), updated_at=_now())
        db.add(assignment)
        db.commit()
        question = AssignmentQuestion(assignment_id=assignment.id, order=0, type="mcq", text="q1",
                                      options=["a", "b"], correct_answer="a", explanation="e",
                                      concept_tag="t", difficulty="easy", user_id=user.id)
        db.add(question)
        db.commit()
        question_id = question.id
        user_id = user.id

    with SessionLocal() as db:
        fetched = db.get(AssignmentQuestion, question_id)
        assert fetched is not None
        assert fetched.user_id == user_id


@pytest.mark.parametrize("status", ["generating", "ready", "failed", "skipped"])
def test_assignment_user_topup_round_trips_each_status(status):
    with SessionLocal() as db:
        user = User(email=f"asg-model-topup-{status}@example.com", password_hash="x")
        db.add(user)
        db.flush()
        _, module_id = _make_chapter_content(db, f"asg-model-topup-{status}")
        assignment = Assignment(level="module", module_id=module_id, scope="global",
                                status="ready", created_at=_now(), updated_at=_now())
        db.add(assignment)
        db.commit()
        topup = AssignmentUserTopup(assignment_id=assignment.id, user_id=user.id, status=status,
                                    created_at=_now(), updated_at=_now())
        db.add(topup)
        db.commit()
        topup_id = topup.id

    with SessionLocal() as db:
        fetched = db.get(AssignmentUserTopup, topup_id)
        assert fetched is not None
        assert fetched.status == status


def test_assignment_user_topup_unique_constraint_rejects_duplicate_pair():
    with SessionLocal() as db:
        user = User(email="asg-model-topup-dup@example.com", password_hash="x")
        db.add(user)
        db.flush()
        _, module_id = _make_chapter_content(db, "asg-model-topup-dup")
        assignment = Assignment(level="module", module_id=module_id, scope="global",
                                status="ready", created_at=_now(), updated_at=_now())
        db.add(assignment)
        db.commit()
        db.add(AssignmentUserTopup(assignment_id=assignment.id, user_id=user.id, status="generating",
                                   created_at=_now(), updated_at=_now()))
        db.commit()
        db.add(AssignmentUserTopup(assignment_id=assignment.id, user_id=user.id, status="generating",
                                   created_at=_now(), updated_at=_now()))
        with pytest.raises(IntegrityError):
            db.commit()
