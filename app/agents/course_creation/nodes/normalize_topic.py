import re
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from app.config import settings
from app.llm.factory import get_chat_model, embed
from app.llm.prompts import CANONICALIZE_TOPIC_PROMPT
from app.llm.retry import call_with_retry
from app.models.course import Course, CourseJob
from app.agents.course_creation.state import CourseCreationState

def _slugify(title: str, max_length: int = 200) -> str:
    """Unicode-aware slug: keeps non-ASCII letters/digits so non-English course
    titles don't collapse into empty or colliding slugs."""
    slug = re.sub(r"[\W_]+", "-", title.strip().lower()).strip("-")
    return slug[:max_length].rstrip("-")

def _find_matching_course(embedding: list[float], db: Session) -> Course | None:
    closest = db.scalar(
        select(Course).order_by(Course.topic_embedding.cosine_distance(embedding)).limit(1)
    )
    if closest is None:
        return None
    distance = db.scalar(
        select(Course.topic_embedding.cosine_distance(embedding)).where(Course.id == closest.id)
    )
    # closest.id was just matched by the query above, so this second lookup
    # against the same exact row is guaranteed to return a value.
    assert distance is not None
    similarity = 1 - distance
    return closest if similarity >= settings.topic_similarity_threshold else None

def find_existing(embedding: list[float], db: Session) -> Course | CourseJob | None:
    course = _find_matching_course(embedding, db)
    if course is not None:
        return course

    job = db.scalar(
        select(CourseJob)
        .where(CourseJob.status.in_(["pending", "running"]))
        .order_by(CourseJob.topic_embedding.cosine_distance(embedding))
        .limit(1)
    )
    if job is not None:
        distance = db.scalar(
            select(CourseJob.topic_embedding.cosine_distance(embedding)).where(CourseJob.id == job.id)
        )
        # job.id was just matched by the query above, so this second lookup
        # against the same exact row is guaranteed to return a value.
        assert distance is not None
        similarity = 1 - distance
        if similarity >= settings.topic_similarity_threshold:
            return job

    return None

def _canonicalize(topic_raw: str) -> str:
    model = get_chat_model("canonicalize_topic")
    messages = CANONICALIZE_TOPIC_PROMPT.format_messages(topic_raw=topic_raw)
    resp = call_with_retry(model.invoke, messages)
    content = resp.content
    # BaseMessage.content is typed as str | list[str | dict] to cover
    # multimodal/tool-call responses; a plain chat completion always returns
    # a str.
    if not isinstance(content, str):
        raise TypeError(f"expected str content from LLM response, got {type(content)}")
    return content.strip()

def normalize_topic(state: CourseCreationState, db: Session) -> CourseCreationState:
    slug = state["topic_slug"]
    embedding = state["topic_embedding"]

    # Production path: the route canonicalized + embedded before enqueueing, so
    # dedup runs on the canonical embedding with no extra LLM/embedding calls.
    if embedding is not None:
        closest = _find_matching_course(embedding, db)
        if closest is not None:
            return {**state, "existing_course_id": closest.id}
        return {**state, "topic_slug": slug, "topic_embedding": embedding,
                "existing_course_id": state["existing_course_id"]}

    # Fallback (legacy/pre-canonicalized job or test seed): cheap raw-embedding
    # scan first so an exact match short-circuits without an LLM call, then
    # canonicalize and re-check on the canonical embedding.
    closest = _find_matching_course(embed(state["topic_raw"]), db)
    if closest is not None:
        return {**state, "existing_course_id": closest.id}

    canonical_title = _canonicalize(state["topic_raw"])
    slug = _slugify(canonical_title)
    embedding = embed(canonical_title)
    closest = _find_matching_course(embedding, db)
    if closest is not None:
        return {**state, "topic_slug": slug, "topic_embedding": embedding,
                "existing_course_id": closest.id}
    return {**state, "topic_slug": slug, "topic_embedding": embedding,
            "existing_course_id": state["existing_course_id"]}
