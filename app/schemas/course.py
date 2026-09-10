from datetime import datetime
from pydantic import BaseModel


class CreateCourseRequest(BaseModel):
    topic: str
    # force=True skips the similarity preview and schedules a new course even
    # when similar courses exist, reusing the cached canonicalization from the
    # preview (search_token). Exact-slug duplicates still attach.
    force: bool = False
    search_token: str | None = None


class CourseCandidate(BaseModel):
    id: int
    topic_slug: str
    topic_raw: str
    similarity: float
    status: str
    course_url: str | None = None
    module_count: int | None = None
    chapter_count: int | None = None


class CourseJobResponse(BaseModel):
    status: str
    job_id: int | None = None
    course: dict | None = None
    error: str | None = None
    search_token: str | None = None
    candidates: list[CourseCandidate] | None = None


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


class CreateExtensionRequest(BaseModel):
    message: str


class ExtensionChapterResult(BaseModel):
    chapter_id: int
    title: str
    objective: str


class ExtensionJobResponse(BaseModel):
    status: str
    job_id: int | None = None
    error: str | None = None
    added: list[ExtensionChapterResult] | None = None


class ExtensionChapterSummary(BaseModel):
    id: int
    title: str
    objective: str
    order: int
    content_ready: bool
