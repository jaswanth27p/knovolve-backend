from pydantic import BaseModel


class AssignmentQuestionResponse(BaseModel):
    id: int
    order: int
    type: str
    text: str
    options: list[str] | None
    # correct_answer and explanation are deliberately NOT exposed here: this
    # endpoint serves the learner the questions to answer, and shipping the
    # answer key with them would defeat the assignment. They stay in the DB as
    # inputs for evaluation.
    concept_tag: str
    difficulty: str


class AssignmentResponse(BaseModel):
    status: str
    questions: list[AssignmentQuestionResponse] | None = None
    error: str | None = None
