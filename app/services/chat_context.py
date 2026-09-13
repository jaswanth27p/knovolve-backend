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


def _load_course_modules(db: Session, course_id: int, user_id: int) -> list[ModuleSummary]:
    """Active course's visible modules + chapters in a constant number of
    queries (module query, chapter query, one batched pass-status query)."""
    modules = (
        db.query(Module)
        .filter(*courses.visible_module_filter(course_id, user_id))
        .order_by(Module.order)
        .all()
    )
    module_ids = [m.id for m in modules]
    chapters = (
        db.query(Chapter).filter(Chapter.module_id.in_(module_ids)).order_by(Chapter.order).all()
        if module_ids else []
    )
    chapters_by_module: dict[int, list[Chapter]] = {}
    for chapter in chapters:
        chapters_by_module.setdefault(chapter.module_id, []).append(chapter)
    passed = progression.passed_chapter_ids(db, user_id, [c.id for c in chapters])

    result: list[ModuleSummary] = []
    for module in modules:
        module_chapters = chapters_by_module.get(module.id, [])
        if module.scope == "user" and not module_chapters:
            continue  # never surface an empty extension bucket
        result.append(ModuleSummary(
            id=module.id, title=module.title,
            chapters=[
                ChapterSummary(id=c.id, title=c.title, completed=c.id in passed)
                for c in module_chapters
            ],
        ))
    return result


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
        # The dashboard already serialized this learner's courses; reuse that
        # row when possible instead of re-querying Course/UserCourse.
        matched = next(
            (r for r in [*dashboard.in_progress, *dashboard.completed] if r.topic_slug == course_slug),
            None,
        )
        if matched is not None:
            current_course = CourseSummary(
                topic_slug=matched.topic_slug, topic_raw=matched.topic_raw,
                status=matched.status, progress=matched.progress,
                course_url=_course_url(matched.topic_slug),
            )
            statuses = get_concept_statuses(db, user_id, matched.id)
            weak = sorted(name for name, status in statuses.items() if status == "weak")
            strong = sorted(name for name, status in statuses.items() if status == "strong")
            current_course_modules = _load_course_modules(db, matched.id, user_id)
        else:
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
                    current_course_modules = _load_course_modules(db, course.id, user_id)

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
