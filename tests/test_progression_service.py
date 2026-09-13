from datetime import datetime, timezone
from app.db import SessionLocal
from app.models.user import User
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.models.assignment import Assignment
from app.models.attempt import AssignmentAttempt
from app.models.enrollment import UserCourse
from app.services import progression


def _now():
    return datetime.now(timezone.utc)


def _make_course_with_chapters(db, slug: str, n_chapters: int):
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
        chapters.append((chapter, assignment))
    return course, chapters


def test_progress_is_fraction_of_passed_chapters():
    with SessionLocal() as db:
        course, chapters = _make_course_with_chapters(db, "prog-a", 2)
        user = User(email="prog-a@example.com", password_hash="x")
        db.add(user); db.flush()
        db.add(UserCourse(user_id=user.id, course_id=course.id, status="in_progress", progress=0.0,
                           enrolled_at=_now(), last_opened_at=_now()))
        db.add(AssignmentAttempt(assignment_id=chapters[0][1].id, user_id=user.id, status="graded",
                                  overall_score=0.9, created_at=_now(), updated_at=_now()))
        db.add(AssignmentAttempt(assignment_id=chapters[1][1].id, user_id=user.id, status="graded",
                                  overall_score=0.3, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, course_id = user.id, course.id

    progression.update_course_progress(db=SessionLocal(), user_id=user_id, course_id=course_id)

    with SessionLocal() as db:
        uc = db.query(UserCourse).filter_by(user_id=user_id, course_id=course_id).one()
        assert uc.progress == 0.5
        assert uc.status == "in_progress"


def test_progress_reaching_one_marks_completed_and_never_reverts():
    with SessionLocal() as db:
        course, chapters = _make_course_with_chapters(db, "prog-b", 1)
        user = User(email="prog-b@example.com", password_hash="x")
        db.add(user); db.flush()
        db.add(UserCourse(user_id=user.id, course_id=course.id, status="in_progress", progress=0.0,
                           enrolled_at=_now(), last_opened_at=_now()))
        db.add(AssignmentAttempt(assignment_id=chapters[0][1].id, user_id=user.id, status="graded",
                                  overall_score=1.0, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, course_id, assignment_id = user.id, course.id, chapters[0][1].id

    with SessionLocal() as db:
        progression.update_course_progress(db, user_id, course_id)
    with SessionLocal() as db:
        uc = db.query(UserCourse).filter_by(user_id=user_id, course_id=course_id).one()
        assert uc.progress == 1.0
        assert uc.status == "completed"

    # A later, worse attempt must not un-complete the course (design doc §6).
    with SessionLocal() as db:
        db.add(AssignmentAttempt(assignment_id=assignment_id, user_id=user_id, status="graded",
                                  overall_score=0.1, created_at=_now(), updated_at=_now()))
        db.commit()
    with SessionLocal() as db:
        progression.update_course_progress(db, user_id, course_id)
    with SessionLocal() as db:
        uc = db.query(UserCourse).filter_by(user_id=user_id, course_id=course_id).one()
        assert uc.status == "completed"


def test_chapter_pass_is_permanent_once_true():
    """Design doc 05: `chapter_passed` is permanent once true, and mastery
    (spec 02) relies on that — a later, worse attempt on the same chapter
    must not flip it back to unpassed."""
    with SessionLocal() as db:
        course, chapters = _make_course_with_chapters(db, "prog-c", 1)
        chapter, assignment = chapters[0]
        user = User(email="prog-c@example.com", password_hash="x")
        db.add(user); db.flush()
        db.add(UserCourse(user_id=user.id, course_id=course.id, status="in_progress", progress=0.0,
                           enrolled_at=_now(), last_opened_at=_now()))
        # earlier passing attempt, later failing attempt
        db.add(AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                  overall_score=1.0, created_at=_now(), updated_at=_now()))
        db.commit()
        import time; time.sleep(0.01)
        db.add(AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="graded",
                                  overall_score=0.2, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, course_id, chapter_id = user.id, course.id, chapter.id

    with SessionLocal() as db:
        progression.update_course_progress(db, user_id, course_id)
    with SessionLocal() as db:
        uc = db.query(UserCourse).filter_by(user_id=user_id, course_id=course_id).one()
        assert uc.progress == 1.0
        assert uc.status == "completed"
        assert progression._chapter_passed(db, chapter_id, user_id) is True


def test_passed_chapter_ids_batches_match_per_chapter():
    with SessionLocal() as db:
        course, chapters = _make_course_with_chapters(db, "prog-e", 2)
        user = User(email="prog-e@example.com", password_hash="x")
        db.add(user); db.flush()
        db.add(AssignmentAttempt(assignment_id=chapters[0][1].id, user_id=user.id, status="graded",
                                  overall_score=0.9, created_at=_now(), updated_at=_now()))
        db.add(AssignmentAttempt(assignment_id=chapters[1][1].id, user_id=user.id, status="graded",
                                  overall_score=0.2, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id = user.id
        ids = [chapters[0][0].id, chapters[1][0].id]

    with SessionLocal() as db:
        got = progression.passed_chapter_ids(db, user_id, ids)
        assert got == {ids[0]}
        for cid in ids:
            assert (cid in got) is progression._chapter_passed(db, cid, user_id)


def test_chapter_passed_uses_users_latest_user_scoped_version_not_global():
    with SessionLocal() as db:
        course, chapters = _make_course_with_chapters(db, "prog-d", 1)
        chapter, global_assignment = chapters[0]
        user = User(email="prog-d@example.com", password_hash="x")
        db.add(user); db.flush()
        db.add(UserCourse(user_id=user.id, course_id=course.id, status="in_progress", progress=0.0,
                           enrolled_at=_now(), last_opened_at=_now()))
        # Global V1 attempt: failed.
        db.add(AssignmentAttempt(assignment_id=global_assignment.id, user_id=user.id, status="graded",
                                  overall_score=0.1, created_at=_now(), updated_at=_now()))
        db.flush()
        # A V2 remediation version + its own user-scoped assignment.
        v2_content = ChapterContent(chapter_id=chapter.id, version=2, scope="user", user_id=user.id,
                                     status="ready", outline=[], created_at=_now(), updated_at=_now())
        db.add(v2_content); db.flush()
        v2_assignment = Assignment(level="chapter", chapter_content_id=v2_content.id, scope="user",
                                    user_id=user.id, status="ready", created_at=_now(), updated_at=_now())
        db.add(v2_assignment); db.flush()
        # V2 attempt: passed.
        db.add(AssignmentAttempt(assignment_id=v2_assignment.id, user_id=user.id, status="graded",
                                  overall_score=0.95, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, course_id = user.id, course.id

    with SessionLocal() as db:
        progression.update_course_progress(db, user_id, course_id)
    with SessionLocal() as db:
        uc = db.query(UserCourse).filter_by(user_id=user_id, course_id=course_id).one()
        assert uc.progress == 1.0
        assert uc.status == "completed"
