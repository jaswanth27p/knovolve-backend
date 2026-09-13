import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_SECTION_QUESTIONS_PROMPT
from app.llm.retry import call_with_retry


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


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

    @field_validator("correct_answer")
    @classmethod
    def _correct_answer_matches_an_option(cls, v: str, info) -> str:
        """Gradeable-by-construction: the answer key must be reachable from the
        UI. Grading compares the selected option text to `correct_answer`
        exactly, so an mcq `correct_answer` that isn't verbatim one of the
        options makes the question impossible. Repair it to the closest option
        (exact → normalized → max token overlap); never reject the whole
        assignment over a formatting mismatch."""
        qtype = info.data.get("type")
        options = info.data.get("options")
        if qtype == "mcq" and options:
            if v in options:
                return v
            normalized = {o.strip().casefold(): o for o in options}
            if v.strip().casefold() in normalized:
                return normalized[v.strip().casefold()]
            answer_tokens = _tokens(v)
            best = max(options, key=lambda o: len(answer_tokens & _tokens(o)))
            return best
        if qtype == "true_false":
            low = v.strip().casefold()
            if low in ("true", "false"):
                return low
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
    if not isinstance(result, SectionQuestionsResponse):
        raise TypeError(f"Unexpected structured output for section questions: {type(result)!r}")
    return result.questions
