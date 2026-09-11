from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.chat import ChatTurn, MAX_HISTORY_TURNS, MAX_MESSAGE_CHARS


class CreateExportRequest(BaseModel):
    kind: Literal["course", "assignments", "full_course", "full_assignments", "custom"]
    params: dict[str, Any] | None = None


class ExportJobResponse(BaseModel):
    id: int
    kind: str
    status: str
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    result_size: int | None = None


class CourseReadinessResponse(BaseModel):
    status: str
    global_content_ready: bool
    additional_content_ready: bool
    versions_ready: bool
    assignments_ready: bool


class GenerationRunResponse(BaseModel):
    id: int
    status: str
    total_units: int
    completed_units: int
    unit_states: list[dict[str, Any]] = []
    error: str | None = None
    action: str | None = None


class ExportPlan(BaseModel):
    title: str
    output_kind: Literal["summary", "qa", "cheat_sheet", "custom"]
    length: Literal["short", "medium", "long"]
    item_count: int | None = None
    notes: str | None = None


class ClarifyRequest(BaseModel):
    message: str = Field(..., max_length=MAX_MESSAGE_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class ClarifyResponse(BaseModel):
    type: Literal["clarifying", "plan"]
    reply: str
    questions: list[str] = []
    plan: ExportPlan | None = None
