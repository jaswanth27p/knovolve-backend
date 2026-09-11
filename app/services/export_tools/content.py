from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from app.models.chapter_content import ChapterContent, ChapterContentSection
from app.models.course import Chapter, Module
from app.services import progression
from app.services.chapter_content import _visible_to_user
from app.services.export_tools._authz import require_export_course
from app.services.export_tools._errors import ExportToolError

MAX_BATCH_CHAPTERS = 20


def _require_chapter(db: Session, course_id: int, chapter_id: int) -> Chapter:
    chapter = db.get(Chapter, chapter_id)
    if chapter is None:
        raise ExportToolError(f"No chapter {chapter_id}.")
    module = db.get(Module, chapter.module_id)
    if module is None or module.course_id != course_id:
        raise ExportToolError(f"Chapter {chapter_id} is not part of this course.")
    return chapter


def _chapter_scope(chapter: Chapter, user_id: int) -> str:
    if chapter.scope == "global":
        return "global"
    if chapter.scope == "user" and chapter.user_id == user_id:
        return "additional"
    raise ExportToolError(f"Chapter {chapter.id} is not available to you.")


def list_modules(db: Session, user_id: int, course_slug: str) -> list[dict]:
    course = require_export_course(db, user_id, course_slug)
    modules = db.query(Module).filter(
        Module.course_id == course.id,
        or_(
            Module.scope == "global",
            and_(Module.scope == "user", Module.user_id == user_id),
        ),
    ).order_by(Module.order).all()
    return [
        {
            "id": m.id,
            "title": m.title,
            "objective": m.objective,
            "scope": "additional" if m.scope == "user" else "global",
        }
        for m in modules
    ]


def list_chapters(db: Session, user_id: int, course_slug: str, module_id: int) -> list[dict]:
    course = require_export_course(db, user_id, course_slug)
    module = db.get(Module, module_id)
    if module is None or module.course_id != course.id:
        raise ExportToolError(f"No module {module_id} in this course.")
    if module.scope == "user" and module.user_id != user_id:
        raise ExportToolError(f"Module {module_id} is not available to you.")
    chapters = db.query(Chapter).filter_by(module_id=module.id).order_by(Chapter.order).all()
    return [
        {"id": c.id, "title": c.title, "objective": c.objective, "scope": _chapter_scope(c, user_id)}
        for c in chapters
    ]


def list_all_chapters(db: Session, user_id: int, course_slug: str) -> list[dict]:
    course = require_export_course(db, user_id, course_slug)
    modules = db.query(Module).filter(
        Module.course_id == course.id,
        or_(
            Module.scope == "global",
            and_(Module.scope == "user", Module.user_id == user_id),
        ),
    ).order_by(Module.order).all()
    out = []
    for module in modules:
        chapters = db.query(Chapter).filter_by(module_id=module.id).order_by(Chapter.order).all()
        for chapter in chapters:
            out.append({
                "chapter_id": chapter.id,
                "module_id": module.id,
                "module_title": module.title,
                "module_scope": "additional" if module.scope == "user" else "global",
                "title": chapter.title,
                "objective": chapter.objective,
            })
    return out


def get_chapter_versions(db: Session, user_id: int, course_slug: str, chapter_id: int) -> dict:
    course = require_export_course(db, user_id, course_slug)
    chapter = _require_chapter(db, course.id, chapter_id)
    scope = _chapter_scope(chapter, user_id)
    versions = db.scalars(
        select(ChapterContent)
        .where(ChapterContent.chapter_id == chapter.id, _visible_to_user(user_id))
        .order_by(ChapterContent.version)
    ).all()
    return {
        "chapter_id": chapter.id,
        "title": chapter.title,
        "scope": scope,
        "version_count": len(versions),
        "versions": [
            {
                "version": v.version,
                "status": v.status,
                "scope": "remediation" if v.remediation_source_attempt_id is not None
                else "additional" if v.scope == "user" else "global",
            }
            for v in versions
        ],
    }


def _serialize_version_content(content: ChapterContent, db: Session) -> dict:
    sections = db.scalars(
        select(ChapterContentSection)
        .where(ChapterContentSection.chapter_content_id == content.id)
        .order_by(ChapterContentSection.order)
    ).all()
    return {
        "available": True,
        "version": content.version,
        "scope": "remediation" if content.remediation_source_attempt_id is not None
        else "additional" if content.scope == "user" else "global",
        "sections": [
            {
                "heading": s.heading,
                "kind": s.kind,
                "body_markdown": s.body_markdown,
                "examples": s.examples or [],
                "diagram_status": s.diagram_status,
                "diagram_image_url": s.diagram_image_url,
            }
            for s in sections
        ],
    }


def get_chapter_version_content(
    db: Session, user_id: int, course_slug: str, chapter_id: int, version: int | None = None,
) -> dict:
    course = require_export_course(db, user_id, course_slug)
    chapter = _require_chapter(db, course.id, chapter_id)
    _chapter_scope(chapter, user_id)
    if version is None:
        content = progression._resolve_relevant_content(db, chapter_id, user_id)
    else:
        content = db.scalar(
            select(ChapterContent).where(
                ChapterContent.chapter_id == chapter.id,
                ChapterContent.version == version,
                _visible_to_user(user_id),
            )
        )
        if content is None:
            raise ExportToolError(f"No version {version} of this chapter is available to you.")
    if content is None or content.status != "ready":
        return {"available": False, "reason": "Chapter content has not been generated yet."}
    return _serialize_version_content(content, db)


def get_chapters_content(
    db: Session, user_id: int, course_slug: str, chapter_ids: list[int], version: int | None = None,
) -> list[dict]:
    if len(chapter_ids) > MAX_BATCH_CHAPTERS:
        raise ExportToolError(f"Fetch at most {MAX_BATCH_CHAPTERS} chapters in one call.")
    return [
        {
            "chapter_id": chapter_id,
            "content": get_chapter_version_content(db, user_id, course_slug, chapter_id, version),
        }
        for chapter_id in chapter_ids
    ]
