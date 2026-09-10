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
    # True for force-created jobs: normalize_topic must skip semantic dedup and
    # build the course even when similar ones exist.
    allow_duplicate: NotRequired[bool]
    # Validation failures seen per retry target, keyed by node name. Must be a
    # real state channel (not a router-local dict): LangGraph rebuilds the state
    # mapping from channel values on every step, so only values *returned* by a
    # node survive across graph steps.
    retry_counts: NotRequired[dict[str, int]]
    # Module titles whose chapters must be regenerated after a cross-module
    # duplicate-title validation failure. Consumed (and reset) by
    # generate_chapters.
    rerun_modules: NotRequired[list[str]]
