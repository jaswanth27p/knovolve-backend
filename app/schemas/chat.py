from pydantic import BaseModel


class ChatRequest(BaseModel):
    course_slug: str | None = None
    chapter_id: int | None = None
    message: str


class ChatResponse(BaseModel):
    reply: str
