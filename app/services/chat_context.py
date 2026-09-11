"""Builds the coarse, once-per-conversation learner context bundle the
chat agent's system prompt is seeded with. Deliberately shallow — it exists
to make the first LLM call cheap, not to answer deep questions; those go
through app.services.chat_tools instead."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.course import Chapter, Course, Module
from app.models.enrollment import UserCourse
from app.schemas.chat import (
    ChapterSummary, CourseSummary, LearnerContextBundle, ModuleSummary, RouteContext,
)
from app.services import courses, progression
from app.services.mastery import get_concept_statuses
from app.services.tracking import get_dashboard


def _course_url(topic_slug: str) -> str:
    return f"{settings.frontend_url}/courses/{topic_slug}"


def build_context_bundle(db: Session, user_id: int, route: RouteContext | None) -> LearnerContextBundle:
    dashboard = get_dashboard(db, user_id)
    my_courses = [
        CourseSummary(
            topic_slug=r.topic_slug, topic_raw=r.topic_raw, status=r.status,
            progress=r.progress, course_url=_course_url(r.topic_slug),
        )
        for r in [*dashboard.in_progress, *dashboard.completed]
    ]

    current_course: CourseSummary | None = None
    current_course_modules: list[ModuleSummary] | None = None
    weak: list[str] = []
    strong: list[str] = []

    course_slug = route.course_slug if route is not None else None
    if course_slug is not None:
        course = db.scalar(select(Course).where(Course.topic_slug == course_slug))
        if course is not None:
            enrollment = db.scalar(
                select(UserCourse).where(UserCourse.user_id == user_id, UserCourse.course_id == course.id)
            )
            current_course = CourseSummary(
                topic_slug=course.topic_slug, topic_raw=course.topic_raw,
                status=enrollment.status if enrollment is not None else "not_started",
                progress=enrollment.progress if enrollment is not None else 0.0,
                course_url=_course_url(course.topic_slug),
            )
            if enrollment is not None:
                statuses = get_concept_statuses(db, user_id, course.id)
                weak = sorted(name for name, status in statuses.items() if status == "weak")
                strong = sorted(name for name, status in statuses.items() if status == "strong")
                modules = db.query(Module).filter(*courses.visible_module_filter(course.id, user_id)).order_by(Module.order).all()
                current_course_modules = []
                for m in modules:
                    chapters = db.query(Chapter).filter_by(module_id=m.id).order_by(Chapter.order).all()
                    if m.scope == "user" and not chapters:
                        continue  # never surface an empty extension bucket
                    current_course_modules.append(
                        ModuleSummary(
                            id=m.id, title=m.title,
                            chapters=[
                                ChapterSummary(
                                    id=c.id, title=c.title,
                                    completed=progression._chapter_passed(db, c.id, user_id),
                                )
                                for c in chapters
                            ],
                        )
                    )

    return LearnerContextBundle(
        in_progress_count=dashboard.in_progress_count,
        completed_count=dashboard.completed_count,
        streak_current=dashboard.streak.current,
        my_courses=my_courses,
        current_course=current_course,
        current_course_modules=current_course_modules,
        weak_concepts=weak,
        strong_concepts=strong,
    )
