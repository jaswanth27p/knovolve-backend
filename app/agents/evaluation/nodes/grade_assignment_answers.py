from pydantic import BaseModel, Field
from app.llm.factory import get_chat_model
from app.llm.prompts import GRADE_ASSIGNMENT_ANSWERS_PROMPT
from app.llm.retry import call_with_retry


class FreeTextAnswerItem(BaseModel):
    question_id: int
    question_text: str
    correct_answer: str
    explanation: str
    concept_tag: str
    user_answer: str


class KnownAnswerItem(BaseModel):
    """An mcq/true_false answer already graded deterministically — passed in
    as context only, never re-graded by this call."""
    question_id: int
    question_text: str
    concept_tag: str
    user_answer: str
    is_correct: bool


class AnswerGrade(BaseModel):
    question_id: int
    is_correct: bool
    feedback: str
    misconception_tag: str | None = None


class GradingResponse(BaseModel):
    grades: list[AnswerGrade] = Field(default_factory=list)
    remediation_concept_tags: list[str] = Field(default_factory=list)
    verdict_reasoning: str


def _free_text_text(items: list[FreeTextAnswerItem]) -> str:
    if not items:
        return "(none — grade nothing here, but still reason over the already-graded questions below for the holistic verdict)"
    return "\n\n".join(
        f"Question {item.question_id} [{item.concept_tag}]: {item.question_text}\n"
        f"Model answer: {item.correct_answer}\n"
        f"Explanation: {item.explanation}\n"
        f"Learner's answer: {item.user_answer}"
        for item in items
    )


def _known_text(items: list[KnownAnswerItem]) -> str:
    if not items:
        return "(none)"
    return "\n\n".join(
        f"Question {item.question_id} [{item.concept_tag}]: {item.question_text}\n"
        f"Learner's answer: {item.user_answer}\n"
        f"Already graded: {'correct' if item.is_correct else 'incorrect'}"
        for item in items
    )


def grade_assignment_answers(
    free_text_items: list[FreeTextAnswerItem], known_answers: list[KnownAnswerItem],
) -> GradingResponse:
    model = get_chat_model("grade_assignment_answers")
    structured = model.with_structured_output(GradingResponse)
    messages = GRADE_ASSIGNMENT_ANSWERS_PROMPT.format_messages(
        free_text_text=_free_text_text(free_text_items), free_text_count=len(free_text_items),
        known_text=_known_text(known_answers),
    )
    result = call_with_retry(structured.invoke, messages)
    return result if isinstance(result, GradingResponse) else GradingResponse.model_validate(result)
