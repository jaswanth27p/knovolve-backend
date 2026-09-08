from pydantic import BaseModel


class SubmitAnswer(BaseModel):
    question_id: int
    answer: str


class SubmitAttemptRequest(BaseModel):
    answers: list[SubmitAnswer]


class AttemptSubmitResponse(BaseModel):
    attempt_id: int
    status: str


class AnswerResult(BaseModel):
    question_id: int
    is_correct: bool
    feedback: str


class ConceptScore(BaseModel):
    concept_tag: str
    correct: int
    total: int


class AttemptResponse(BaseModel):
    status: str
    overall_score: float | None = None
    answers: list[AnswerResult] | None = None
    concept_scores: list[ConceptScore] | None = None
    error: str | None = None
