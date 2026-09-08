from pydantic import BaseModel


class AssignmentQuestionResponse(BaseModel):
    id: int
    order: int
    type: str
    text: str
    options: list[str] | None
    correct_answer: str
    explanation: str
    concept_tag: str
    difficulty: str


class AssignmentResponse(BaseModel):
    status: str
    questions: list[AssignmentQuestionResponse] | None = None
    error: str | None = None
