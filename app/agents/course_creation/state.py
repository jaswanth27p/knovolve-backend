from typing import NotRequired, TypedDict, Optional


class CourseCreationState(TypedDict):
    job_id: int
    topic_raw: str
    topic_slug: Optional[str]
    topic_embedding: Optional[list[float]]
    existing_course_id: Optional[int]
    modules: Optional[list]
    concepts: Optional[list]
    concept_edges: Optional[list]
    error: Optional[str]
    # Validation failures seen per retry target, keyed by node name. Must be a
    # real state channel (not a router-local dict): LangGraph rebuilds the state
    # mapping from channel values on every step, so only values *returned* by a
    # node survive across graph steps.
    retry_counts: NotRequired[dict[str, int]]
