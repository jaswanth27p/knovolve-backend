import uuid
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from app.db import SessionLocal
from app.models.course import Course, Module, Chapter, Concept, ConceptEdge
from app.agents.course_creation.state import CourseCreationState
from app.agents.course_creation.nodes.persist_course import persist_course
from app.agents.course_creation.nodes.generate_outline import ModuleDraft
from app.agents.course_creation.nodes.generate_chapters import ChapterDraft
from app.agents.course_creation.nodes.build_concept_graph import (
    ConceptDraft,
    ConceptEdgeDraft,
    ConceptGraphResponse,
)
from app.agents.course_creation.graph import (
    MAX_RETRIES_PER_NODE,
    CourseGenerationError,
    CoursePersistenceError,
    build_course_creation_graph,
)
from app.agents.course_creation import graph as graph_mod


def test_persist_course_writes_full_tree():
    state: CourseCreationState = {
        "job_id": 1, "topic_raw": "TypeScript", "topic_slug": "ts-persist-test",
        "topic_embedding": [0.0] * 1024, "existing_course_id": None,
        "modules": [{"title": "M1", "objective": "o", "order": 1,
                     "chapters": [{"title": "C1", "objective": "o", "order": 1}]}],
        "concepts": [{"name": "X", "chapter_title": "C1"}],
        "concept_edges": [], "error": None,
    }
    with SessionLocal() as db:
        result = persist_course(state, db)
        db.commit()

    assert result["error"] is None
    with SessionLocal() as db:
        course = db.query(Course).filter_by(topic_slug="ts-persist-test").one()
        assert db.query(Module).filter_by(course_id=course.id).count() == 1
        module = db.query(Module).filter_by(course_id=course.id).one()
        assert db.query(Chapter).filter_by(module_id=module.id).count() == 1
        assert db.query(Concept).filter_by(course_id=course.id).count() == 1


def test_persist_course_resolves_string_keys_to_real_fks():
    """Task 8 emits concepts/edges keyed by *name*/*chapter_title* strings;
    persist_course must resolve them to real DB ids."""
    state: CourseCreationState = {
        "job_id": 2, "topic_raw": "Rust", "topic_slug": "rust-fk-test",
        "topic_embedding": [0.1] * 1024, "existing_course_id": None,
        "modules": [
            {"title": "M1", "objective": "o", "order": 1, "chapters": [
                {"title": "Ownership", "objective": "o", "order": 1},
                {"title": "Borrowing", "objective": "o", "order": 2},
            ]},
        ],
        "concepts": [
            {"name": "Move semantics", "chapter_title": "Ownership"},
            {"name": "Borrow checker", "chapter_title": "Borrowing"},
        ],
        "concept_edges": [
            {"concept_name": "Borrow checker", "prerequisite_name": "Move semantics"},
        ],
        "error": None,
    }
    with SessionLocal() as db:
        result = persist_course(state, db)
        db.commit()

    assert result["error"] is None
    assert result["existing_course_id"] is not None

    with SessionLocal() as db:
        course = db.query(Course).filter_by(topic_slug="rust-fk-test").one()
        assert result["existing_course_id"] == course.id

        ownership = db.query(Chapter).filter_by(title="Ownership").one()
        borrowing = db.query(Chapter).filter_by(title="Borrowing").one()

        move = db.query(Concept).filter_by(course_id=course.id, name="Move semantics").one()
        borrow = db.query(Concept).filter_by(course_id=course.id, name="Borrow checker").one()
        # string chapter_title resolved to the correct chapter row
        assert move.chapter_id == ownership.id
        assert borrow.chapter_id == borrowing.id

        edge = db.query(ConceptEdge).one()
        assert edge.concept_id == borrow.id
        assert edge.prerequisite_concept_id == move.id


def test_persist_course_is_atomic_on_bad_reference():
    """An edge naming a concept that does not exist must abort the whole write:
    no course, no modules, no chapters left behind."""
    state: CourseCreationState = {
        "job_id": 3, "topic_raw": "Go", "topic_slug": "go-atomic-test",
        "topic_embedding": [0.2] * 1024, "existing_course_id": None,
        "modules": [{"title": "M1", "objective": "o", "order": 1,
                     "chapters": [{"title": "C1", "objective": "o", "order": 1}]}],
        "concepts": [{"name": "Goroutines", "chapter_title": "C1"}],
        "concept_edges": [
            {"concept_name": "Goroutines", "prerequisite_name": "Does Not Exist"},
        ],
        "error": None,
    }
    with SessionLocal() as db:
        result = persist_course(state, db)
        db.commit()

    assert result["error"] is not None

    with SessionLocal() as db:
        assert db.query(Course).filter_by(topic_slug="go-atomic-test").count() == 0
        assert db.query(Module).count() == 0
        assert db.query(Chapter).count() == 0
        assert db.query(Concept).count() == 0


def test_persist_course_rejects_duplicate_chapter_titles_across_modules():
    """Concepts reference chapters by title alone, so two modules sharing a
    chapter title makes every concept naming it ambiguous. That must be a loud
    failure, not a silently mis-attached FK."""
    state: CourseCreationState = {
        "job_id": 4, "topic_raw": "Python", "topic_slug": "py-collision-test",
        "topic_embedding": [0.3] * 1024, "existing_course_id": None,
        "modules": [
            {"title": "M1", "objective": "o", "order": 1, "chapters": [
                {"title": "Introduction", "objective": "o", "order": 1},
            ]},
            {"title": "M2", "objective": "o", "order": 2, "chapters": [
                {"title": "Introduction", "objective": "o", "order": 1},
            ]},
        ],
        "concepts": [{"name": "Ambiguous", "chapter_title": "Introduction"}],
        "concept_edges": [],
        "error": None,
    }
    with SessionLocal() as db:
        result = persist_course(state, db)
        db.commit()

    assert result["error"] is not None
    assert "duplicate chapter title" in result["error"]
    assert "Introduction" in result["error"]

    # Rolled back before returning: nothing reached the DB despite the
    # add()/flush() calls made along the way.
    with SessionLocal() as db:
        assert db.query(Course).filter_by(topic_slug="py-collision-test").count() == 0
        assert db.query(Module).count() == 0
        assert db.query(Chapter).count() == 0
        assert db.query(Concept).count() == 0


# --------------------------------------------------------------------------
# Full-graph integration
# --------------------------------------------------------------------------


#: pgvector's cosine_distance is undefined for the zero vector, so the fake
#: embedding must be non-zero for the dedup path to be exercised at all.
FAKE_EMBEDDING = [1.0] + [0.0] * 1023


def _patched_graph(outline, chapters, concept_graph, canonical_title,
                   embedding=FAKE_EMBEDDING):
    """Patch every LLM/embedding boundary the graph touches. No real API calls."""

    outline_model = MagicMock()
    outline_model.with_structured_output.return_value.invoke.return_value = outline
    chapters_model = MagicMock()
    chapters_model.with_structured_output.return_value.invoke.return_value = chapters
    graph_model = MagicMock()
    graph_model.with_structured_output.return_value.invoke.return_value = concept_graph

    by_node = {
        "generate_outline": outline_model,
        "generate_chapters": chapters_model,
        "build_concept_graph": graph_model,
    }

    stack = ExitStack()
    stack.enter_context(
        patch("app.agents.course_creation.nodes.normalize_topic.embed",
              return_value=embedding)
    )
    stack.enter_context(
        patch("app.agents.course_creation.nodes.normalize_topic._canonicalize",
              return_value=canonical_title)
    )
    for node in by_node:
        stack.enter_context(
            patch(f"app.agents.course_creation.nodes.{node}.get_chat_model",
                  side_effect=lambda n: by_node[n])
        )
    return stack, by_node


def _initial_state(job_id: int, topic_raw: str) -> CourseCreationState:
    return {
        "job_id": job_id, "topic_raw": topic_raw, "topic_slug": None,
        "topic_embedding": None, "existing_course_id": None,
        "modules": None, "concepts": None, "concept_edges": None, "error": None,
    }


def test_full_graph_topic_to_persisted_course():
    stack, _ = _patched_graph(
        outline=[ModuleDraft(title="Basics", objective="o", order=1)],
        chapters=[ChapterDraft(title="Intro", objective="o", order=1)],
        concept_graph=ConceptGraphResponse(
            concepts=[ConceptDraft(name="Basics Concept", chapter_title="Intro")],
            edges=[],
        ),
        canonical_title="Full Graph Test Topic",
    )
    with stack:
        app_graph = build_course_creation_graph()
        final_state = app_graph.invoke(
            _initial_state(999, "full graph test topic"),
            config={"configurable": {"thread_id": f"test-{uuid.uuid4()}"}},
        )

    assert final_state["error"] is None
    assert final_state["existing_course_id"] is not None

    with SessionLocal() as db:
        course = db.query(Course).filter_by(topic_slug="full-graph-test-topic").one()
        assert final_state["existing_course_id"] == course.id
        module = db.query(Module).filter_by(course_id=course.id).one()
        assert module.title == "Basics"
        chapter = db.query(Chapter).filter_by(module_id=module.id).one()
        assert chapter.title == "Intro"
        concept = db.query(Concept).filter_by(course_id=course.id).one()
        assert concept.name == "Basics Concept"
        assert concept.chapter_id == chapter.id


def test_full_graph_short_circuits_on_existing_course():
    """normalize_topic finding a near-duplicate must end the run without
    generating or persisting anything new."""
    with SessionLocal() as db:
        dup_state: CourseCreationState = {
            "job_id": 0, "topic_raw": "Dup", "topic_slug": "dup-course",
            "topic_embedding": FAKE_EMBEDDING, "existing_course_id": None,
            "modules": [], "concepts": [], "concept_edges": [], "error": None,
        }
        existing = persist_course(dup_state, db)
        db.commit()
    existing_id = existing["existing_course_id"]

    stack, by_node = _patched_graph(
        outline=[], chapters=[],
        concept_graph=ConceptGraphResponse(concepts=[], edges=[]),
        canonical_title="Should Not Be Used",
    )
    with stack:
        app_graph = build_course_creation_graph()
        final_state = app_graph.invoke(
            _initial_state(1000, "dup"),
            config={"configurable": {"thread_id": f"test-{uuid.uuid4()}"}},
        )

    assert final_state["existing_course_id"] == existing_id
    by_node["generate_outline"].with_structured_output.assert_not_called()
    with SessionLocal() as db:
        assert db.query(Course).count() == 1


# --------------------------------------------------------------------------
# Retry cap
#
# The plan's original `_route_after_validate` incremented `_retry_counts` by
# mutating the state dict inside the conditional-edge function. Verified
# against langgraph 1.2.11: those mutations are discarded (the state mapping is
# rebuilt from channel values each step), so the cap never tripped and the
# graph looped until GraphRecursionError. Counting now happens in the
# validate_course node and is returned as a state update. These tests pin that.
# --------------------------------------------------------------------------


def test_retry_cap_trips_after_exactly_max_retries():
    """A module that never gets chapters must fail validation, retry
    generate_chapters exactly MAX_RETRIES_PER_NODE times, then raise."""
    stack, by_node = _patched_graph(
        outline=[ModuleDraft(title="Basics", objective="o", order=1)],
        chapters=[],  # always empty -> validate_course always errors
        concept_graph=ConceptGraphResponse(concepts=[], edges=[]),
        canonical_title="Retry Cap Test",
    )
    with stack:
        app_graph = build_course_creation_graph()
        with pytest.raises(CourseGenerationError, match="exhausted retries"):
            app_graph.invoke(
                _initial_state(1001, "retry cap test"),
                config={"configurable": {"thread_id": f"test-{uuid.uuid4()}"}},
            )

        chapters_invoke = by_node["generate_chapters"].with_structured_output.return_value.invoke
        # 1 initial attempt + MAX_RETRIES_PER_NODE retries, then give up.
        assert chapters_invoke.call_count == 1 + MAX_RETRIES_PER_NODE

    with SessionLocal() as db:
        assert db.query(Course).count() == 0


def test_retry_count_persists_across_graph_cycles():
    """Directly pin the failure mode the plan's router had: the counter must
    actually increment across cycles rather than resetting each step."""
    from app.agents.course_creation import graph as graph_mod

    observed: list[dict] = []
    real_router = graph_mod._route_after_validate

    def spy(state):
        observed.append(dict(state.get("retry_counts") or {}))
        return real_router(state)

    stack, _ = _patched_graph(
        outline=[ModuleDraft(title="Basics", objective="o", order=1)],
        chapters=[],
        concept_graph=ConceptGraphResponse(concepts=[], edges=[]),
        canonical_title="Retry Count Persist",
    )
    with stack, patch.object(graph_mod, "_route_after_validate", spy):
        app_graph = build_course_creation_graph()
        with pytest.raises(CourseGenerationError):
            app_graph.invoke(
                _initial_state(1002, "retry count persist"),
                config={"configurable": {"thread_id": f"test-{uuid.uuid4()}"}},
            )

    counts = [c.get("generate_chapters", 0) for c in observed]
    assert counts == [1, 2, 3], f"retry counter did not persist across cycles: {counts}"


def test_recoverable_failure_retries_then_succeeds():
    """One bad concept-graph response followed by a good one must recover
    without tripping the cap."""
    bad = ConceptGraphResponse(
        concepts=[ConceptDraft(name="A", chapter_title="Intro")],
        edges=[ConceptEdgeDraft(concept_name="A", prerequisite_name="Ghost")],
    )
    good = ConceptGraphResponse(
        concepts=[ConceptDraft(name="A", chapter_title="Intro")], edges=[]
    )
    stack, by_node = _patched_graph(
        outline=[ModuleDraft(title="Basics", objective="o", order=1)],
        chapters=[ChapterDraft(title="Intro", objective="o", order=1)],
        concept_graph=good,
        canonical_title="Recoverable Retry",
    )
    with stack:
        by_node["build_concept_graph"].with_structured_output.return_value.invoke.side_effect = [
            bad,
            good,
        ]
        app_graph = build_course_creation_graph()
        final_state = app_graph.invoke(
            _initial_state(1003, "recoverable retry"),
            config={"configurable": {"thread_id": f"test-{uuid.uuid4()}"}},
        )

    assert final_state["error"] is None
    assert final_state["retry_counts"] == {"build_concept_graph": 1}
    with SessionLocal() as db:
        course = db.query(Course).filter_by(topic_slug="recoverable-retry").one()
        assert db.query(Concept).filter_by(course_id=course.id).count() == 1


def test_retry_target_matches_exact_validate_course_templates():
    """Pin the retry-target check against validate_course's three real error
    shapes, including a concept name that contains the word 'chapter'."""
    from app.agents.course_creation.graph import _retry_target

    assert _retry_target("module 'M1' has no chapters") == "generate_chapters"
    assert _retry_target(
        "duplicate chapter title 'Introduction' found more than once across the course"
    ) == "generate_chapters"
    assert _retry_target(
        "concept_edge references unknown concept: "
        "{'concept_name': 'Intro Chapter Concepts', 'prerequisite_name': 'Ghost'}"
    ) == "build_concept_graph"
    assert _retry_target(
        "concept 'X' references unknown chapter 'Never Exists'"
    ) == "build_concept_graph"
    assert _retry_target("concept prerequisite graph is cyclic") == "build_concept_graph"


# --------------------------------------------------------------------------
# Persist failures must raise, never silently succeed (issue 1)
# --------------------------------------------------------------------------


def test_persist_wrapper_raises_on_error_state_and_does_not_commit():
    """The graph node wrapper must turn persist_course's error-carrying return
    value into a raised CoursePersistenceError — a silent error-state return is
    exactly what let a failed persist commit the job as 'succeeded' with
    course_id=NULL."""
    fake_session = MagicMock()
    state = _initial_state(1, "persist wrapper test")
    with patch.object(graph_mod, "persist_course",
                      return_value={**state, "error": "boom"}), \
         patch.object(graph_mod, "SessionLocal", return_value=fake_session):
        with pytest.raises(CoursePersistenceError, match="boom"):
            graph_mod._persist_course_db(state)
    fake_session.commit.assert_not_called()


def test_full_graph_raises_when_persist_fails():
    """End-to-end: even with otherwise valid content, a persist failure must
    surface as a raised exception out of graph.invoke, not a state with
    error set (which the task would have committed as success)."""
    stack, _ = _patched_graph(
        outline=[ModuleDraft(title="Basics", objective="o", order=1)],
        chapters=[ChapterDraft(title="Intro", objective="o", order=1)],
        concept_graph=ConceptGraphResponse(
            concepts=[ConceptDraft(name="C", chapter_title="Intro")], edges=[]
        ),
        canonical_title="Persist Raise Test",
    )
    with stack:
        app_graph = build_course_creation_graph()
        with patch.object(graph_mod, "persist_course",
                          return_value={**_initial_state(1010, "persist raise"),
                                        "error": "persist boom"}):
            with pytest.raises(CoursePersistenceError, match="persist boom"):
                app_graph.invoke(
                    _initial_state(1010, "persist raise"),
                    config={"configurable": {"thread_id": f"test-{uuid.uuid4()}"}},
                )


def test_full_graph_repairs_duplicate_chapter_titles_via_targeted_rerun():
    """A cross-module duplicate chapter title is caught by validate_course,
    routed back to generate_chapters with only the offending modules flagged,
    regenerated with uniqueness constraints, and the course then persists."""
    stack, by_node = _patched_graph(
        outline=[
            ModuleDraft(title="Basics", objective="o", order=1),
            ModuleDraft(title="Advanced", objective="o", order=2),
        ],
        chapters=[],
        concept_graph=ConceptGraphResponse(
            concepts=[
                ConceptDraft(name="Intro Concept", chapter_title="Intro"),
                ConceptDraft(name="Opt Concept", chapter_title="Optimization"),
            ],
            edges=[],
        ),
        canonical_title="Dup Repair Test",
    )
    gen = by_node["generate_chapters"].with_structured_output.return_value.invoke
    # Pass 1: both modules emit "Intro" (collision). Closure of generate_outline
    # keeps the outline; only the two flagged modules are regenerated on pass 2,
    # with the second module told to avoid the first module's titles.
    gen.side_effect = [
        [ChapterDraft(title="Intro", objective="o", order=1)],
        [ChapterDraft(title="Intro", objective="o", order=1)],
        [ChapterDraft(title="Intro", objective="o", order=1)],
        [ChapterDraft(title="Optimization", objective="o", order=1)],
    ]

    with stack:
        app_graph = build_course_creation_graph()
        final_state = app_graph.invoke(
            _initial_state(1011, "dup repair"),
            config={"configurable": {"thread_id": f"test-{uuid.uuid4()}"}},
        )

    assert final_state["error"] is None
    assert final_state["existing_course_id"] is not None
    assert gen.call_count == 4
    with SessionLocal() as db:
        course = db.query(Course).filter_by(topic_slug="dup-repair-test").one()
        module_ids = [m.id for m in db.query(Module).filter_by(course_id=course.id).all()]
        assert len(module_ids) == 2
        all_titles = {
            ch.title
            for ch in db.query(Chapter).filter(Chapter.module_id.in_(module_ids)).all()
        }
        assert all_titles == {"Intro", "Optimization"}
