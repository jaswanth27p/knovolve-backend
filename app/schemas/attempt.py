from pydantic import BaseModel


class SubmitAnswer(BaseModel):
    question_id: int
    answer: str


class SubmitAttemptRequest(BaseModel):
    answers: list[SubmitAnswer]


class AttemptSubmitResponse(BaseModel):
    attempt_id: int
    status: str
