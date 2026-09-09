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


def test_uses_latest_attempt_per_chapter_not_best():
    with SessionLocal() as db:
        course, chapters = _make_course_with_chapters(db, "prog-c", 1)
        user = User(email="prog-c@example.com", password_hash="x")
        db.add(user); db.flush()
        db.add(UserCourse(user_id=user.id, course_id=course.id, status="in_progress", progress=0.0,
                           enrolled_at=_now(), last_opened_at=_now()))
        # earlier good attempt, later bad attempt -> "latest" wins, chapter not passed
        db.add(AssignmentAttempt(assignment_id=chapters[0][1].id, user_id=user.id, status="graded",
                                  overall_score=1.0, created_at=_now(), updated_at=_now()))
        db.commit()
        import time; time.sleep(0.01)
        db.add(AssignmentAttempt(assignment_id=chapters[0][1].id, user_id=user.id, status="graded",
                                  overall_score=0.2, created_at=_now(), updated_at=_now()))
        db.commit()
        user_id, course_id = user.id, course.id

    with SessionLocal() as db:
        progression.update_course_progress(db, user_id, course_id)
    with SessionLocal() as db:
        uc = db.query(UserCourse).filter_by(user_id=user_id, course_id=course_id).one()
        assert uc.progress == 0.0
