from datetime import datetime
from pydantic import BaseModel


class CreateCourseRequest(BaseModel):
    topic: str


class CourseJobResponse(BaseModel):
    status: str
    job_id: int | None = None
    course: dict | None = None
    error: str | None = None


class MyCourseJobResponse(BaseModel):
    id: int
    topic_slug: str
    topic_raw: str
    status: str
    error: str | None = None
    course_slug: str | None = None
    created_at: datetime


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
    weak_concept_count: int
    strong_concept_count: int


class PaginatedTrackedCoursesResponse(BaseModel):
    items: list[TrackedCourseResponse]
    total: int
    page: int
    limit: int
    total_pages: int


class StreakResponse(BaseModel):
    current: int
    longest: int


class ActivityEvent(BaseModel):
    at: datetime          # tz-aware UTC
    score: float          # overall_score, 0.0-1.0 (graded rows always have one)
    passed: bool          # score >= progression.PASS_THRESHOLD

class ActivityResponse(BaseModel):
    days: int             # requested window (default 14)
    events: list[ActivityEvent]


class DashboardResponse(BaseModel):
    in_progress: list[TrackedCourseResponse]
    completed: list[TrackedCourseResponse]
    in_progress_count: int
    completed_count: int
    total_count: int
    streak: StreakResponse


class PublicCourseResponse(BaseModel):
    id: int
    topic_slug: str
    topic_raw: str
    created_at: datetime
    module_count: int
    chapter_count: int


class PaginatedPublicCoursesResponse(BaseModel):
    items: list[PublicCourseResponse]
    total: int
    page: int
    limit: int
    total_pages: int


class ChapterVersionSummary(BaseModel):
    version: int
    status: str
    created_at: datetime
    remediation_target_tags: list[str] | None = None


class ChapterContentSectionResponse(BaseModel):
    order: int
    heading: str
    kind: str
    body_markdown: str
    examples: list[dict]
    diagram_status: str | None = None
    diagram_image_url: str | None = None


class ChapterVersionDetail(ChapterVersionSummary):
    error: str | None = None
    sections: list[ChapterContentSectionResponse]
