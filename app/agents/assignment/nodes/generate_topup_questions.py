from app.llm.factory import get_chat_model
from app.llm.prompts import GENERATE_TOPUP_QUESTIONS_PROMPT
from app.llm.retry import call_with_retry
from app.agents.assignment.nodes.generate_section_questions import QuestionDraft, SectionQuestionsResponse


def _sections_text(sections: list[dict]) -> str:
    return "\n\n".join(f"## {s['heading']}\n{s['body_markdown']}" for s in sections)


def generate_topup_questions(
    title: str, objective: str, sections: list[dict], count: int,
) -> list[QuestionDraft]:
    model = get_chat_model("generate_topup_questions")
    structured = model.with_structured_output(SectionQuestionsResponse)
    messages = GENERATE_TOPUP_QUESTIONS_PROMPT.format_messages(
        title=title, objective=objective, sections_text=_sections_text(sections), count=count,
    )
    result = call_with_retry(structured.invoke, messages)
    return result.questions if isinstance(result, SectionQuestionsResponse) else result
