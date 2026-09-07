import re
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from app.llm.factory import get_chat_model, embed
from app.llm.prompts import CANONICALIZE_TOPIC_PROMPT
from app.models.course import Course
from app.agents.course_creation.state import CourseCreationState

SIMILARITY_THRESHOLD = 0.85

def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

def _canonicalize(topic_raw: str) -> str:
    model = get_chat_model("canonicalize_topic")
    messages = CANONICALIZE_TOPIC_PROMPT.format_messages(topic_raw=topic_raw)
    resp = model.invoke(messages)
    content = resp.content
    # BaseMessage.content is typed as str | list[str | dict] to cover
    # multimodal/tool-call responses; a plain chat completion always returns
    # a str.
    if not isinstance(content, str):
        raise TypeError(f"expected str content from LLM response, got {type(content)}")
    return content.strip()

def normalize_topic(state: CourseCreationState, db: Session) -> CourseCreationState:
    embedding = embed(state["topic_raw"])

    closest = db.scalar(
        select(Course)
        .order_by(Course.topic_embedding.cosine_distance(embedding))
        .limit(1)
    )
    if closest is not None:
        distance = db.scalar(
            select(Course.topic_embedding.cosine_distance(embedding)).where(Course.id == closest.id)
        )
        # closest.id was just matched by the query above, so this second
        # lookup against the same exact row is guaranteed to return a value.
        assert distance is not None
        similarity = 1 - distance
        if similarity >= SIMILARITY_THRESHOLD:
            return {**state, "existing_course_id": closest.id, "topic_embedding": embedding}

    canonical_title = _canonicalize(state["topic_raw"])
    return {
        **state,
        "topic_slug": _slugify(canonical_title),
        "topic_embedding": embedding,
        "existing_course_id": None,
    }
