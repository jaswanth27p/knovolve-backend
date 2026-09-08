from pydantic import BaseModel, Field
from app.llm.factory import get_chat_model
from app.llm.prompts import GRADE_FREE_TEXT_ANSWERS_PROMPT
from app.llm.retry import call_with_retry


class FreeTextAnswerItem(BaseModel):
    question_id: int
    question_text: str
    correct_answer: str
    explanation: str
    user_answer: str


class AnswerGrade(BaseModel):
    question_id: int
    is_correct: bool
    feedback: str


class GradingResponse(BaseModel):
    grades: list[AnswerGrade] = Field(default_factory=list)


def _items_text(items: list[FreeTextAnswerItem]) -> str:
    return "\n\n".join(
        f"Question {item.question_id}: {item.question_text}\n"
        f"Model answer: {item.correct_answer}\n"
        f"Explanation: {item.explanation}\n"
        f"Learner's answer: {item.user_answer}"
        for item in items
    )


def grade_free_text_answers(items: list[FreeTextAnswerItem]) -> list[AnswerGrade]:
    model = get_chat_model("grade_free_text_answers")
    structured = model.with_structured_output(GradingResponse)
    messages = GRADE_FREE_TEXT_ANSWERS_PROMPT.format_messages(
        items_text=_items_text(items), count=len(items),
    )
    result = call_with_retry(structured.invoke, messages)
    return result.grades if isinstance(result, GradingResponse) else result
