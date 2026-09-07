from datetime import datetime
from pydantic import BaseModel


class CreateCourseRequest(BaseModel):
    topic: str


class CourseJobResponse(BaseModel):
    status: str
    job_id: int | None = None
    course: dict | None = None
    error: str | None = None


class TrackedCourseResponse(BaseModel):
    id: int
    topic_slug: str
    topic_raw: str
    status: str
    progress: float
    last_opened_at: datetime
    module_count: int
    chapter_count: int
    content_ready: bool


class DashboardResponse(BaseModel):
    in_progress: list[TrackedCourseResponse]
    completed: list[TrackedCourseResponse]
    in_progress_count: int
    completed_count: int
    total_count: int


class PublicCourseResponse(BaseModel):
    id: int
    topic_slug: str
    topic_raw: str
    module_count: int
    chapter_count: int
