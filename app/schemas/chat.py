from typing import Literal

from pydantic import BaseModel, Field

# Client-sent `history`/`message` are echoed back into the prompt on every
# turn, so they are bounded here to keep a single request from inflating LLM
# cost or overflowing the model context window. app.services.chat additionally
# truncates to the most recent MAX_PROMPT_HISTORY_TURNS turns before prompting.
MAX_MESSAGE_CHARS = 8000
MAX_CHAT_TURN_CHARS = 20000
MAX_HISTORY_TURNS = 100


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=MAX_CHAT_TURN_CHARS)


class RouteContext(BaseModel):
    page: Literal["dashboard", "course", "chapter_content", "assignment", "attempt_review"]
    course_slug: str | None = None
    chapter_id: int | None = None
    assignment_id: int | None = None
    attempt_id: int | None = None


class CourseSummary(BaseModel):
    topic_slug: str
    topic_raw: str
    status: str
    progress: float
    course_url: str


class ChapterSummary(BaseModel):
    id: int
    title: str
    completed: bool


class ModuleSummary(BaseModel):
    id: int
    title: str
    chapters: list[ChapterSummary]


class LearnerContextBundle(BaseModel):
    in_progress_count: int
    completed_count: int
    streak_current: int
    my_courses: list[CourseSummary]
    current_course: CourseSummary | None = None
    current_course_modules: list[ModuleSummary] | None = None
    weak_concepts: list[str] = []
    strong_concepts: list[str] = []


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=MAX_MESSAGE_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    context: LearnerContextBundle | None = None
    current_route: RouteContext | None = None


class ChatResponse(BaseModel):
    reply: str
    # Only populated on the turn that built a fresh bundle (context was None
    # in the request) — the client caches it and never gets it echoed back.
    context: LearnerContextBundle | None = None
