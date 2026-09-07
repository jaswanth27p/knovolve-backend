from typing import TypedDict, Optional


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
