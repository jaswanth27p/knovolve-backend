from langgraph.graph import END, StateGraph

from app.agents.course_creation.checkpointer import get_checkpointer
from app.agents.course_creation.nodes.build_concept_graph import build_concept_graph
from app.agents.course_creation.nodes.generate_chapters import generate_chapters
from app.agents.course_creation.nodes.generate_outline import generate_outline
from app.agents.course_creation.nodes.normalize_topic import normalize_topic
from app.agents.course_creation.nodes.persist_course import persist_course
from app.agents.course_creation.nodes.validate_course import validate_course
from app.agents.course_creation.state import CourseCreationState
from app.db import SessionLocal

MAX_RETRIES_PER_NODE = 2


class CourseGenerationError(RuntimeError):
    """Raised when a node keeps failing validation past its retry budget."""


def _retry_target(error: str) -> str:
    """A missing-chapters complaint is fixed by regenerating chapters; every
    other validation failure is about the concept graph."""
    return "generate_chapters" if "chapter" in error else "build_concept_graph"


def _normalize_topic_db(state: CourseCreationState) -> CourseCreationState:
    with SessionLocal() as db:
        return normalize_topic(state, db)


def _persist_course_db(state: CourseCreationState) -> CourseCreationState:
    with SessionLocal() as db:
        result = persist_course(state, db)
        if result["error"] is None:
            db.commit()
        return result


def _validate_course_counting(state: CourseCreationState) -> CourseCreationState:
    """`validate_course` plus retry accounting.

    The retry counter is incremented *here*, in a node, and returned as part of
    the state update — a conditional-edge function cannot do this, because the
    state dict it receives is rebuilt from channel values each step and any
    in-place mutation of it is discarded (verified empirically against
    langgraph 1.2.11).
    """
    result = validate_course(state)
    counts = dict(state.get("retry_counts") or {})
    if result["error"] is not None:
        target = _retry_target(result["error"])
        counts[target] = counts.get(target, 0) + 1
    return {**result, "retry_counts": counts}


def _route_after_normalize(state: CourseCreationState) -> str:
    return END if state["existing_course_id"] else "generate_outline"


def _route_after_validate(state: CourseCreationState) -> str:
    """Pure routing decision — all counting already happened in the node."""
    if state["error"] is None:
        return "persist_course"

    target = _retry_target(state["error"])
    # counts[target] is the number of failures observed so far. The first
    # failure buys retry #1, so we're out of budget once it exceeds the cap.
    failures = (state.get("retry_counts") or {}).get(target, 0)
    if failures > MAX_RETRIES_PER_NODE:
        raise CourseGenerationError(
            f"exhausted retries on {target} after {MAX_RETRIES_PER_NODE} "
            f"attempts: {state['error']}"
        )
    return target


def build_course_creation_graph():
    graph = StateGraph(CourseCreationState)
    graph.add_node("normalize_topic", _normalize_topic_db)
    graph.add_node("generate_outline", generate_outline)
    graph.add_node("generate_chapters", generate_chapters)
    graph.add_node("build_concept_graph", build_concept_graph)
    graph.add_node("validate_course", _validate_course_counting)
    graph.add_node("persist_course", _persist_course_db)

    graph.set_entry_point("normalize_topic")
    graph.add_conditional_edges(
        "normalize_topic", _route_after_normalize, ["generate_outline", END]
    )
    graph.add_edge("generate_outline", "generate_chapters")
    graph.add_edge("generate_chapters", "build_concept_graph")
    graph.add_edge("build_concept_graph", "validate_course")
    graph.add_conditional_edges(
        "validate_course",
        _route_after_validate,
        ["persist_course", "generate_chapters", "build_concept_graph"],
    )
    graph.add_edge("persist_course", END)

    return graph.compile(checkpointer=get_checkpointer())
