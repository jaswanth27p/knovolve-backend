from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.agents.course_creation.state import CourseCreationState
from app.models.course import Chapter, Concept, ConceptEdge, Course, Module


def persist_course(state: CourseCreationState, db: Session) -> CourseCreationState:
    """Write the whole validated course tree in one unit of work.

    Task 8 hands us concepts/edges keyed by *strings* (`chapter_title`,
    `concept_name`, `prerequisite_name`); we resolve those to real row ids as we
    insert. Nothing is committed here — the caller owns the transaction — but on
    any failure we roll back so a partially-flushed tree can never be committed
    by that caller.
    """
    try:
        course = Course(
            topic_slug=state["topic_slug"],
            topic_raw=state["topic_raw"],
            topic_embedding=state["topic_embedding"],
            status="published",
            created_at=datetime.now(timezone.utc),
        )
        db.add(course)
        db.flush()

        chapter_id_by_title: dict[str, int] = {}
        for module_draft in state["modules"]:
            module = Module(
                course_id=course.id,
                title=module_draft["title"],
                objective=module_draft["objective"],
                order=module_draft["order"],
            )
            db.add(module)
            db.flush()
            for chapter_draft in module_draft["chapters"]:
                title = chapter_draft["title"]
                # Concepts reference chapters by *title* only (Task 8's data
                # shape), so two chapters sharing a title anywhere in the course
                # make every concept naming it ambiguous. Silently keeping the
                # last-inserted id would attach concepts to the wrong module's
                # chapter and commit a corrupted graph, so fail loudly instead.
                if title in chapter_id_by_title:
                    raise ValueError(
                        f"duplicate chapter title {title!r} across modules — "
                        f"cannot unambiguously resolve concept references"
                    )
                chapter = Chapter(
                    module_id=module.id,
                    title=title,
                    objective=chapter_draft["objective"],
                    order=chapter_draft["order"],
                )
                db.add(chapter)
                db.flush()
                chapter_id_by_title[title] = chapter.id

        concept_id_by_name: dict[str, int] = {}
        for concept_draft in state["concepts"]:
            chapter_title = concept_draft["chapter_title"]
            if chapter_title not in chapter_id_by_title:
                raise KeyError(f"concept references unknown chapter: {chapter_title!r}")
            concept = Concept(
                course_id=course.id,
                name=concept_draft["name"],
                chapter_id=chapter_id_by_title[chapter_title],
            )
            db.add(concept)
            db.flush()
            concept_id_by_name[concept_draft["name"]] = concept.id

        for edge_draft in state["concept_edges"]:
            for key in ("concept_name", "prerequisite_name"):
                if edge_draft[key] not in concept_id_by_name:
                    raise KeyError(
                        f"concept_edge references unknown concept: {edge_draft[key]!r}"
                    )
            db.add(
                ConceptEdge(
                    concept_id=concept_id_by_name[edge_draft["concept_name"]],
                    prerequisite_concept_id=concept_id_by_name[
                        edge_draft["prerequisite_name"]
                    ],
                )
            )
        db.flush()
    except Exception as exc:  # noqa: BLE001 - surfaced on state["error"] by design
        db.rollback()
        return {**state, "error": f"persist_course failed: {exc}"}

    return {**state, "existing_course_id": course.id, "error": None}
