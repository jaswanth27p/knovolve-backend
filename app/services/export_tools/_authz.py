from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.course import Course
from app.models.enrollment import UserCourse
from app.services.export_tools._errors import ExportToolError


def require_export_course(db: Session, user_id: int, course_slug: str) -> Course:
    course = db.scalar(select(Course).where(Course.topic_slug == course_slug))
    if course is None:
        raise ExportToolError(f"No course found with slug '{course_slug}'.")
    enrollment = db.scalar(
        select(UserCourse).where(UserCourse.user_id == user_id, UserCourse.course_id == course.id)
    )
    if enrollment is None:
        raise ExportToolError(f"You haven't started the course '{course.topic_raw}' yet.")
    return course
