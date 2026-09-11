from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.llm.factory import embed
from app.models.course import Chapter, Course, Module
from app.models.enrollment import UserCourse
from app.services import courses as courses_service
from app.services import progression
from app.services import tracking as tracking_service
from app.services.chat_tools._authz import require_started_course
from app.services.chat_tools._errors import ChatToolError


def _course_url(topic_slug: str) -> str:
    return f"{settings.frontend_url}/courses/{topic_slug}"


def browse_public_courses(db: Session, user_id: int) -> list[dict]:
    rows, _total = courses_service.list_public_courses(db, user_id)
    return [
        {
            "topic_slug": r.topic_slug, "topic_raw": r.topic_raw,
            "module_count": r.module_count, "chapter_count": r.chapter_count,
            "course_url": _course_url(r.topic_slug),
        }
        for r in rows
    ]


def list_my_courses(db: Session, user_id: int) -> list[dict]:
    rows, _total = tracking_service.list_my_courses(db, user_id)
    return [
        {
            "topic_slug": r.topic_slug, "topic_raw": r.topic_raw, "status": r.status,
            "progress": r.progress, "module_count": r.module_count, "chapter_count": r.chapter_count,
            "course_url": _course_url(r.topic_slug),
        }
        for r in rows
    ]


def search_courses(db: Session, user_id: int, query: str, limit: int = 5) -> list[dict]:
    embedding = embed(query)
    rows = db.scalars(
        select(Course).order_by(Course.topic_embedding.cosine_distance(embedding)).limit(limit)
    ).all()
    return [
        {"topic_slug": c.topic_slug, "topic_raw": c.topic_raw, "course_url": _course_url(c.topic_slug)}
        for c in rows
    ]


def get_course_detail(db: Session, user_id: int, course_slug: str) -> dict:
    """Public catalog read — deliberately NOT gated by require_started_course.

    A learner must be able to inspect a course the agent surfaced via
    browse_public_courses/search_courses before starting it. Only this user's
    own enrollment row is used for `started`/`progress`, so nothing private to
    another learner is exposed; deep content stays behind the scoped tools."""
    course = db.scalar(select(Course).where(Course.topic_slug == course_slug))
    if course is None:
        raise ChatToolError(f"No course found with slug '{course_slug}'.")
    modules = db.query(Module).filter(*courses_service.visible_module_filter(course.id, None)).all()
    chapter_count = sum(db.query(Chapter).filter_by(module_id=m.id).count() for m in modules)
    enrollment = db.scalar(
        select(UserCourse).where(UserCourse.user_id == user_id, UserCourse.course_id == course.id)
    )
    return {
        "topic_slug": course.topic_slug, "topic_raw": course.topic_raw,
        "module_count": len(modules), "chapter_count": chapter_count,
        "started": enrollment is not None,
        "progress": enrollment.progress if enrollment is not None else None,
        "course_url": _course_url(course.topic_slug),
    }


def get_course_modules(db: Session, user_id: int, course_slug: str) -> list[dict]:
    course = require_started_course(db, user_id, course_slug)
    modules = db.query(Module).filter(*courses_service.visible_module_filter(course.id, user_id)) \
        .order_by(Module.order).all()
    result = []
    for m in modules:
        chapters = db.query(Chapter).filter_by(module_id=m.id).order_by(Chapter.order).all()
        if m.scope == "user" and not chapters:
            continue  # never surface an empty extension bucket
        result.append({
            "id": m.id, "title": m.title,
            "chapters": [
                {"id": c.id, "title": c.title, "completed": progression._chapter_passed(db, c.id, user_id)}
                for c in chapters
            ],
        })
    return result
