import re
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from app.llm.factory import get_chat_model, embed
from app.models.course import Course
from app.agents.course_creation.state import CourseCreationState

SIMILARITY_THRESHOLD = 0.85

def _slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")

def _canonicalize(topic_raw: str) -> str:
    model = get_chat_model("canonicalize_topic")
    resp = model.invoke(
        f"Rewrite this learning topic as a clean, canonical course title "
        f"(3-6 words, no extra commentary, just the title): {topic_raw}"
    )
    return resp.content.strip()

def normalize_topic(state: CourseCreationState, db: Session) -> CourseCreationState:
    embedding = embed(state["topic_raw"])

    closest = db.scalar(
        select(Course)
        .order_by(Course.topic_embedding.cosine_distance(embedding))
        .limit(1)
    )
    if closest is not None:
        similarity = 1 - db.scalar(
            select(Course.topic_embedding.cosine_distance(embedding)).where(Course.id == closest.id)
        )
        if similarity >= SIMILARITY_THRESHOLD:
            return {**state, "existing_course_id": closest.id, "topic_embedding": embedding}

    canonical_title = _canonicalize(state["topic_raw"])
    return {
        **state,
        "topic_slug": _slugify(canonical_title),
        "topic_embedding": embedding,
        "existing_course_id": None,
    }
