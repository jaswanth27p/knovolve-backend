from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel


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
