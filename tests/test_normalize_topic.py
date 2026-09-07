from datetime import datetime, timezone
from unittest.mock import patch
from app.db import SessionLocal
from app.models.course import Course
from app.agents.course_creation.nodes.normalize_topic import normalize_topic

def test_exact_slug_match_short_circuits():
    with SessionLocal() as db:
        db.add(Course(topic_slug="typescript", topic_raw="TypeScript",
                       topic_embedding=[1.0] + [0.0] * 1023,
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        existing = db.query(Course).filter_by(topic_slug="typescript").one()

    state = {"job_id": 1, "topic_raw": "TypeScript", "topic_slug": None,
              "topic_embedding": None, "existing_course_id": None,
              "modules": None, "concepts": None, "concept_edges": None, "error": None}

    with patch("app.agents.course_creation.nodes.normalize_topic.embed", return_value=[1.0] + [0.0] * 1023):
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] == existing.id

def test_semantic_match_short_circuits_despite_different_wording():
    with SessionLocal() as db:
        db.add(Course(topic_slug="typescript-generics", topic_raw="TypeScript Generics",
                       topic_embedding=[0.99] + [0.0] * 1023,
                       created_at=datetime.now(timezone.utc)))
        db.commit()
        existing = db.query(Course).filter_by(topic_slug="typescript-generics").one()

    state = {"job_id": 2, "topic_raw": "generics in typescript", "topic_slug": None,
              "topic_embedding": None, "existing_course_id": None,
              "modules": None, "concepts": None, "concept_edges": None, "error": None}

    # cosine-similar but not identical vector -> should still match above threshold
    with patch("app.agents.course_creation.nodes.normalize_topic.embed", return_value=[0.98] + [0.0] * 1023):
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] == existing.id

def test_no_match_canonicalizes_and_continues():
    state = {"job_id": 3, "topic_raw": "closures in js", "topic_slug": None,
              "topic_embedding": None, "existing_course_id": None,
              "modules": None, "concepts": None, "concept_edges": None, "error": None}

    with patch("app.agents.course_creation.nodes.normalize_topic.embed", return_value=[0.0] * 1024), \
         patch("app.agents.course_creation.nodes.normalize_topic._canonicalize", return_value="JavaScript Closures"):
        with SessionLocal() as db:
            result = normalize_topic(state, db)
    assert result["existing_course_id"] is None
    assert result["topic_slug"] == "javascript-closures"
    assert result["topic_embedding"] == [0.0] * 1024
