from pydantic import BaseModel


class CreateCourseRequest(BaseModel):
    topic: str


class CourseJobResponse(BaseModel):
    status: str
    job_id: int | None = None
    course: dict | None = None
    error: str | None = None
