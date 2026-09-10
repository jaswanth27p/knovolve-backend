"""Builds this request's tool list: 16 read-only chat-agent tools, each a
closure over the request's `db` session and authenticated `user_id` so the
underlying functions (app.services.chat_tools.{courses,mastery,chapters,
assignments,stats}) never see untrusted identity from the LLM."""
from langchain_core.tools import BaseTool, tool
from sqlalchemy.orm import Session

from app.services.chat_tools import assignments, chapters, courses, mastery, stats


def build_tools(db: Session, user_id: int) -> list[BaseTool]:
    @tool
    def browse_public_courses() -> list[dict]:
        """List public courses the learner has not started yet."""
        return courses.browse_public_courses(db, user_id)

    @tool
    def list_my_courses() -> list[dict]:
        """List courses the learner has started, with progress."""
        return courses.list_my_courses(db, user_id)

    @tool
    def search_courses(query: str) -> list[dict]:
        """Semantically search the course catalog for a topic, e.g. 'search for a python course'."""
        return courses.search_courses(db, user_id, query)

    @tool
    def get_course_detail(course_slug: str) -> dict:
        """Get metadata for one course by its slug."""
        return courses.get_course_detail(db, user_id, course_slug)

    @tool
    def get_course_modules(course_slug: str) -> list[dict]:
        """Get the module/chapter breakdown with completion for a course the learner started."""
        return courses.get_course_modules(db, user_id, course_slug)

    @tool
    def get_weak_concepts_by_course(course_slug: str) -> dict:
        """Get concept mastery status (weak/strong/unassessed) for a whole course."""
        return mastery.get_weak_concepts_by_course(db, user_id, course_slug)

    @tool
    def get_weak_concepts_by_module(course_slug: str, module_id: int) -> dict:
        """Get concept mastery status for one module."""
        return mastery.get_weak_concepts_by_module(db, user_id, course_slug, module_id)

    @tool
    def get_weak_concepts_by_chapter(course_slug: str, chapter_id: int) -> dict:
        """Get concept mastery status for one chapter."""
        return mastery.get_weak_concepts_by_chapter(db, user_id, course_slug, chapter_id)

    @tool
    def get_recurring_weak_concepts(course_slug: str, min_occurrences: int = 3) -> list[dict]:
        """Get concepts that stayed weak across 3+ remediation versions, ranked by how many versions they persisted."""
        return mastery.get_recurring_weak_concepts(db, user_id, course_slug, min_occurrences)

    @tool
    def get_chapter_progress(course_slug: str, chapter_id: int) -> dict:
        """Get a chapter's completion status and how many content versions/attempts the learner has taken."""
        return chapters.get_chapter_progress(db, user_id, course_slug, chapter_id)

    @tool
    def get_chapter_content(course_slug: str, chapter_id: int) -> dict:
        """Fetch the chapter's generated content if it exists. Never triggers generation."""
        return chapters.get_chapter_content(db, user_id, course_slug, chapter_id)

    @tool
    def get_assignment(assignment_id: int) -> dict:
        """Get metadata about an assignment: level, status, which course it belongs to."""
        return assignments.get_assignment(db, user_id, assignment_id)

    @tool
    def list_assignment_attempts(assignment_id: int) -> list[dict]:
        """List the learner's own attempts on an assignment, most recent first."""
        return assignments.list_assignment_attempts(db, user_id, assignment_id)

    @tool
    def get_attempt_detail(attempt_id: int) -> dict:
        """Get per-question breakdown of one of the learner's own attempts: answers, right/wrong, misconceptions."""
        return assignments.get_attempt_detail(db, user_id, attempt_id)

    @tool
    def get_question(question_id: int) -> dict:
        """Get a single assignment question's full detail."""
        return assignments.get_question(db, user_id, question_id)

    @tool
    def get_user_stats() -> dict:
        """Get the learner's stats: streak, in-progress/completed counts, assignments attempted today."""
        return stats.get_user_stats(db, user_id)

    return [
        browse_public_courses, list_my_courses, search_courses, get_course_detail, get_course_modules,
        get_weak_concepts_by_course, get_weak_concepts_by_module, get_weak_concepts_by_chapter,
        get_recurring_weak_concepts,
        get_chapter_progress, get_chapter_content,
        get_assignment, list_assignment_attempts, get_attempt_detail, get_question,
        get_user_stats,
    ]
