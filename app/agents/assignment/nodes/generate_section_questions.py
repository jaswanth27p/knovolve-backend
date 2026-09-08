from typing import Literal
from pydantic import BaseModel, Field, field_validator
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_SECTION_QUESTIONS_PROMPT
from app.llm.retry import call_with_retry


class QuestionDraft(BaseModel):
    type: Literal["mcq", "true_false", "free_text"]
    text: str
    options: list[str] | None = None
    correct_answer: str
    explanation: str
    concept_tag: str
    difficulty: Literal["easy", "medium", "hard"]

    @field_validator("options")
    @classmethod
    def _mcq_requires_options(cls, v: list[str] | None, info) -> list[str] | None:
        if info.data.get("type") == "mcq" and (not v or len(v) < 2):
            raise ValueError("mcq questions require at least 2 options")
        return v


class SectionQuestionsResponse(BaseModel):
    questions: list[QuestionDraft] = Field(default_factory=list)


def _examples_text(examples: list[dict]) -> str:
    if not examples:
        return "(none)"
    return "\n".join(f"- {e['prompt']}: {e['walkthrough']}" for e in examples)


def generate_questions_for_section(
    chapter_title: str, chapter_objective: str, heading: str, body_markdown: str, examples: list[dict],
) -> list[QuestionDraft]:
    model = get_chat_model("generate_section_questions")
    structured = model.with_structured_output(SectionQuestionsResponse)
    messages = GENERATE_SECTION_QUESTIONS_PROMPT.format_messages(
        chapter_title=chapter_title, chapter_objective=chapter_objective,
        heading=heading, body_markdown=body_markdown, examples_text=_examples_text(examples),
    )
    result = call_with_retry(structured.invoke, messages)
    return result.questions if isinstance(result, SectionQuestionsResponse) else result
