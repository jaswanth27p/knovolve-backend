"""Course-scoped read-only tools for custom PDF exports."""
from langchain_core.tools import BaseTool, tool
from sqlalchemy.orm import Session
from app.services.export_tools import assignments, content


def build_export_tools(db: Session, user_id: int, course_slug: str) -> list[BaseTool]:
    @tool
    def list_modules() -> list[dict]:
        """List global and your own Additional Chapters modules."""
        return content.list_modules(db, user_id, course_slug)

    @tool
    def list_chapters(module_id: int) -> list[dict]:
        """List chapters in one module."""
        return content.list_chapters(db, user_id, course_slug, module_id)

    @tool
    def list_all_chapters() -> list[dict]:
        """List every chapter available to you, including Additional Chapters."""
        return content.list_all_chapters(db, user_id, course_slug)

    @tool
    def get_chapter_versions(chapter_id: int) -> dict:
        """List a chapter’s versions and how many versions it has."""
        return content.get_chapter_versions(db, user_id, course_slug, chapter_id)

    @tool
    def get_chapter_version_content(chapter_id: int, version: int | None = None) -> dict:
        """Read one chapter version’s generated markdown; omit version for the newest relevant version."""
        return content.get_chapter_version_content(db, user_id, course_slug, chapter_id, version)

    @tool
    def get_chapters_content(chapter_ids: list[int], version: int | None = None) -> list[dict]:
        """Read generated markdown for up to 20 chapters in one call."""
        return content.get_chapters_content(db, user_id, course_slug, chapter_ids, version)

    @tool
    def list_assignments() -> list[dict]:
        """List chapter and module assignments available to you, with status."""
        return assignments.list_assignments(db, user_id, course_slug)

    @tool
    def get_assignment_questions(assignment_id: int) -> dict:
        """Read an assignment’s questions, including answers and explanations."""
        return assignments.get_assignment_questions(db, user_id, course_slug, assignment_id)

    return [
        list_modules,
        list_chapters,
        list_all_chapters,
        get_chapter_versions,
        get_chapter_version_content,
        get_chapters_content,
        list_assignments,
        get_assignment_questions,
    ]
