from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course
from app.agents.course_creation.state import CourseCreationState
from app.agents.course_creation.nodes.normalize_topic import normalize_topic

def test_exact_slug_match_short_circuits():
    with SessionLocal() as db:
        db.add(Course(topic_slug="typescript", topic_raw="TypeScript",
                       topic_embedding=[1.0] + [0.0] * 2047,
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        existing = db.query(Course).filter_by(topic_slug="typescript").one()

    state: CourseCreationState = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": None,
              "topic_embedding": None, "existing_course_id": None,
              "modules": None, "concepts": None, "concept_edges": None, "error": None}

    with patch("app.agents.course_creation.nodes.normalize_topic.embed", return_value=[1.0] + [0.0] * 2047):
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] == existing.id

def test_semantic_match_short_circuits_despite_different_wording():
    with SessionLocal() as db:
        db.add(Course(topic_slug="typescript-generics", topic_raw="TypeScript Generics",
                       topic_embedding=[0.99] + [0.0] * 2047,
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        existing = db.query(Course).filter_by(topic_slug="typescript-generics").one()

    state: CourseCreationState = {"job_id": 2, "topic_raw": "generics in typescript", "topic_slug": None,
              "topic_embedding": None, "existing_course_id": None,
              "modules": None, "concepts": None, "concept_edges": None, "error": None}

    # cosine-similar but not identical vector -> should still match above threshold
    with patch("app.agents.course_creation.nodes.normalize_topic.embed", return_value=[0.98] + [0.0] * 2047):
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] == existing.id

def test_no_match_canonicalizes_and_continues():
    state: CourseCreationState = {"job_id": 3, "topic_raw": "closures in js", "topic_slug": None,
              "topic_embedding": None, "existing_course_id": None,
              "modules": None, "concepts": None, "concept_edges": None, "error": None}

    with patch("app.agents.course_creation.nodes.normalize_topic.embed", return_value=[0.0] * 2048), \
         patch("app.agents.course_creation.nodes.normalize_topic._canonicalize", return_value="JavaScript Closures"):
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] is None
    assert result["topic_slug"] == "javascript-closures"
    assert result["topic_embedding"] == [0.0] * 2048

def test_preseeded_canonical_embedding_skips_llm():
    """Production path: the route has already canonicalized and embedded, so the
    graph must dedup on that embedding without re-canonicalizing or re-embedding."""
    with SessionLocal() as db:
        db.add(Course(topic_slug="typescript", topic_raw="TypeScript",
                       topic_embedding=[1.0] + [0.0] * 2047,
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        course_id = db.query(Course).filter_by(topic_slug="typescript").one().id

    state: CourseCreationState = {"job_id": 6, "topic_raw": "i want to learn typescript",
              "topic_slug": "typescript", "topic_embedding": [1.0] + [0.0] * 2047,
              "existing_course_id": None, "modules": None, "concepts": None,
              "concept_edges": None, "error": None}

    with patch("app.agents.course_creation.nodes.normalize_topic._canonicalize") as mock_can, \
         patch("app.agents.course_creation.nodes.normalize_topic.embed") as mock_embed:
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] == course_id
    mock_can.assert_not_called()
    mock_embed.assert_not_called()

def test_allow_duplicate_skips_dedup():
    """Force-created jobs bypass semantic dedup entirely: even a near-identical
    existing course must not short-circuit the graph."""
    with SessionLocal() as db:
        db.add(Course(topic_slug="typescript-generics", topic_raw="TypeScript Generics",
                       topic_embedding=[0.99] + [0.0] * 2047,
                       created_at=datetime.now(timezone.utc)))
        db.commit()

    state: CourseCreationState = {"job_id": 9, "topic_raw": "generics in typescript",
              "topic_slug": "generics-in-typescript", "topic_embedding": [0.98] + [0.0] * 2047,
              "existing_course_id": None, "modules": None, "concepts": None,
              "concept_edges": None, "error": None, "allow_duplicate": True}

    with patch("app.agents.course_creation.nodes.normalize_topic.embed") as mock_embed, \
         patch("app.agents.course_creation.nodes.normalize_topic._canonicalize") as mock_can:
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] is None
    assert result["topic_slug"] == "generics-in-typescript"
    assert result["topic_embedding"] == [0.98] + [0.0] * 2047
    mock_can.assert_not_called()
    mock_embed.assert_not_called()

def test_slugify_preserves_unicode_and_separators():
    from app.agents.course_creation.nodes.normalize_topic import _slugify
    assert _slugify("日本語 Programming") == "日本語-programming"
    assert _slugify("TypeScript 2.0") == "typescript-2-0"
    assert _slugify("C++") == "c"

def test_similarity_threshold_reads_from_settings(monkeypatch):
    import math
    from app.config import settings
    with SessionLocal() as db:
        db.add(Course(topic_slug="typescript", topic_raw="TypeScript",
                       topic_embedding=[1.0] + [0.0] * 2047,
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        course_id = db.query(Course).filter_by(topic_slug="typescript").one().id

    text = "TypeScript"
    near = [0.9, math.sqrt(1 - 0.9 ** 2)] + [0.0] * 2046  # cos-sim 0.9 with the course vector
    state: CourseCreationState = {"job_id": 5, "topic_raw": text, "topic_slug": None,
              "topic_embedding": None, "existing_course_id": None,
              "modules": None, "concepts": None, "concept_edges": None, "error": None}

    monkeypatch.setattr(settings, "topic_similarity_threshold", 0.80)
    with patch("app.agents.course_creation.nodes.normalize_topic.embed", return_value=near):
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] == course_id

    monkeypatch.setattr(settings, "topic_similarity_threshold", 0.95)
    with patch("app.agents.course_creation.nodes.normalize_topic.embed", return_value=near), \
         patch("app.agents.course_creation.nodes.normalize_topic._canonicalize", return_value=text):
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] is None
    assert result["topic_slug"] == "typescript"
