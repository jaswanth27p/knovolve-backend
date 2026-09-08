import json
from datetime import datetime, timezone
from typing import Iterator
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.db import get_session
from app.services import chapter_content, courses
from app.services.enrollment import touch_enrollment
from app.auth.dependencies import get_current_user
from app.models.course import Course, Module, Chapter
from app.models.chapter_content import ChapterContent
from app.models.assignment import Assignment, AssignmentQuestion
from app.models.attempt import AssignmentAttempt, AssignmentAnswer
from app.schemas.course import CreateCourseRequest, CourseJobResponse, PublicCourseResponse
from app.schemas.assignment import AssignmentQuestionResponse, AssignmentResponse
from app.schemas.attempt import SubmitAttemptRequest, AttemptSubmitResponse, AnswerResult, ConceptScore, AttemptResponse
from app.tasks.assignment_tasks import generate_chapter_assignment_task, generate_module_assignment_task
from app.tasks.evaluation_tasks import grade_assignment_attempt_task

router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("", response_model=CourseJobResponse, status_code=202)
def create_course(body: CreateCourseRequest, response: Response, db: Session = Depends(get_session),
                  user=Depends(get_current_user)):
    result = courses.create_course_job(db, user.id, body.topic)
    if result.status == "exists":
        response.status_code = 200
    return result


@router.get("/jobs/{job_id}", response_model=CourseJobResponse)
def get_job(job_id: int, db: Session = Depends(get_session), user=Depends(get_current_user)):
    return courses.get_course_job(db, job_id)


@router.post("/jobs/{job_id}/retry", response_model=CourseJobResponse, status_code=202)
def retry_job(job_id: int, response: Response, db: Session = Depends(get_session),
              user=Depends(get_current_user)):
    result = courses.retry_course_job(db, job_id)
    if result.status == "exists":
        response.status_code = 200
    return result


@router.get("", response_model=list[PublicCourseResponse])
def list_public_courses(db: Session = Depends(get_session), user=Depends(get_current_user)):
    return courses.list_public_courses(db, user.id)


@router.get("/{slug}")
def get_course(slug: str, db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    touch_enrollment(db, user.id, course)
    return courses.serialize_course(db, course)


@router.get("/{slug}/chapters/{chapter_id}/content")
def get_chapter_content(slug: str, chapter_id: int, db: Session = Depends(get_session),
                        user=Depends(get_current_user)):
    course = courses.get_course_by_slug(db, slug)
    chapter = chapter_content.get_chapter(db, course, chapter_id)
    touch_enrollment(db, user.id, course)

    def _encode(events: Iterator[dict]) -> Iterator[str]:
        for event in events:
            yield json.dumps(event) + "\n"

    return StreamingResponse(
        _encode(chapter_content.stream_content_events(chapter, db)),
        media_type="application/x-ndjson",
    )


def _chapter_assignment(db: Session, chapter_content_id: int) -> Assignment | None:
    """The global chapter assignment for this content, if any. Filtered on
    level+scope the same way `_get_or_create_assignment` filters: the schema
    allows a per-user row on the same chapter_content_id, and an unfiltered
    single-row fetch would then be free to hand one learner another's
    assignment. Scope stays hardcoded to "global" — V1 only generates those."""
    return db.scalar(
        select(Assignment).where(
            Assignment.level == "chapter",
            Assignment.scope == "global",
            Assignment.chapter_content_id == chapter_content_id,
        )
    )


def _module_assignment(db: Session, module_id: int) -> Assignment | None:
    """The global module assignment for this module, if any. See
    `_chapter_assignment` for why level/scope are filtered."""
    return db.scalar(
        select(Assignment).where(
            Assignment.level == "module",
            Assignment.scope == "global",
            Assignment.module_id == module_id,
        )
    )


def _serialize_assignment(assignment: Assignment, db: Session) -> AssignmentResponse:
    if assignment.status != "ready":
        return AssignmentResponse(status=assignment.status, error=assignment.error)
    questions = db.scalars(
        select(AssignmentQuestion)
        .where(AssignmentQuestion.assignment_id == assignment.id)
        .order_by(AssignmentQuestion.order)
    ).all()
    return AssignmentResponse(
        status="ready",
        questions=[
            AssignmentQuestionResponse(
                id=q.id, order=q.order, type=q.type, text=q.text, options=q.options,
                concept_tag=q.concept_tag, difficulty=q.difficulty,
            )
            for q in questions
        ],
    )


@router.get("/{slug}/chapters/{chapter_id}/assignment", response_model=AssignmentResponse)
def get_chapter_assignment(slug: str, chapter_id: int, db: Session = Depends(get_session),
                            user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    chapter = (
        db.query(Chapter)
        .join(Module, Chapter.module_id == Module.id)
        .filter(Chapter.id == chapter_id, Module.course_id == course.id)
        .first()
    )
    if not chapter:
        raise HTTPException(status_code=404, detail="chapter not found")

    content = db.scalar(
        select(ChapterContent).where(ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global")
    )
    if content is None or content.status != "ready":
        raise HTTPException(status_code=404, detail="chapter content not ready")

    assignment = _chapter_assignment(db, content.id)
    if assignment is None or assignment.status == "failed":
        generate_chapter_assignment_task.delay(content.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = _chapter_assignment(db, content.id)
    if assignment is None:
        return AssignmentResponse(status="generating")
    return _serialize_assignment(assignment, db)


def _module_chapters_ready(module: Module, db: Session) -> bool:
    chapters = db.query(Chapter).filter_by(module_id=module.id).all()
    if not chapters:
        return False
    for chapter in chapters:
        content = db.scalar(
            select(ChapterContent).where(ChapterContent.chapter_id == chapter.id, ChapterContent.scope == "global")
        )
        if content is None or content.status != "ready":
            return False
    return True


@router.post("/{slug}/modules/{module_id}/assignment", response_model=AssignmentResponse, status_code=202)
def create_module_assignment(slug: str, module_id: int, response: Response,
                              db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    module = db.query(Module).filter_by(id=module_id, course_id=course.id).first()
    if not module:
        raise HTTPException(status_code=404, detail="module not found")
    if not _module_chapters_ready(module, db):
        raise HTTPException(status_code=409, detail="not all chapters in this module have ready content")

    assignment = _module_assignment(db, module.id)
    if assignment is None or assignment.status == "failed":
        generate_module_assignment_task.delay(module.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = _module_assignment(db, module.id)
    if assignment is None:
        return AssignmentResponse(status="generating")
    response.status_code = 200 if assignment.status == "ready" else 202
    return _serialize_assignment(assignment, db)


@router.get("/{slug}/modules/{module_id}/assignment", response_model=AssignmentResponse)
def get_module_assignment(slug: str, module_id: int, db: Session = Depends(get_session),
                           user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    module = db.query(Module).filter_by(id=module_id, course_id=course.id).first()
    if not module:
        raise HTTPException(status_code=404, detail="module not found")

    assignment = _module_assignment(db, module.id)
    if assignment is None or assignment.status == "failed":
        if not _module_chapters_ready(module, db):
            raise HTTPException(status_code=409, detail="not all chapters in this module have ready content")
        generate_module_assignment_task.delay(module.id)  # pyright: ignore[reportFunctionMemberAccess]
        assignment = _module_assignment(db, module.id)
    if assignment is None:
        return AssignmentResponse(status="generating")
    return _serialize_assignment(assignment, db)


def _assignment_belongs_to_course(db: Session, assignment: Assignment, course_id: int) -> bool:
    """True if `assignment` (chapter- or module-level) is part of `course_id`.
    Used by both the submit and fetch attempt routes so an assignment id from
    a different course can never be submitted/fetched against this slug."""
    if assignment.level == "chapter":
        content = db.get(ChapterContent, assignment.chapter_content_id)
        if content is None:
            return False
        chapter = db.get(Chapter, content.chapter_id)
        if chapter is None:
            return False
        module = db.get(Module, chapter.module_id)
        return module is not None and module.course_id == course_id
    module = db.get(Module, assignment.module_id)
    return module is not None and module.course_id == course_id


@router.post("/{slug}/assignments/{assignment_id}/attempts", response_model=AttemptSubmitResponse, status_code=202)
def submit_assignment_attempt(slug: str, assignment_id: int, body: SubmitAttemptRequest,
                               db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    assignment = db.scalar(
        select(Assignment).where(Assignment.id == assignment_id, Assignment.scope == "global")
    )
    if assignment is None or assignment.status != "ready" or not _assignment_belongs_to_course(db, assignment, course.id):
        raise HTTPException(status_code=404, detail="assignment not found")

    questions = db.scalars(
        select(AssignmentQuestion).where(AssignmentQuestion.assignment_id == assignment.id)
    ).all()
    questions_by_id = {q.id: q for q in questions}
    submitted_ids = {a.question_id for a in body.answers}
    if len(body.answers) != len(submitted_ids):
        raise HTTPException(status_code=400, detail="duplicate question_id in submitted answers")
    if submitted_ids != set(questions_by_id.keys()):
        raise HTTPException(status_code=400, detail="submitted answers must cover exactly the assignment's questions")

    now = datetime.now(timezone.utc)
    attempt = AssignmentAttempt(assignment_id=assignment.id, user_id=user.id, status="grading",
                                 created_at=now, updated_at=now)
    db.add(attempt)
    db.flush()
    for a in body.answers:
        question = questions_by_id[a.question_id]
        db.add(AssignmentAnswer(attempt_id=attempt.id, question_id=question.id,
                                 concept_tag=question.concept_tag, user_answer=a.answer))
    db.commit()

    grade_assignment_attempt_task.delay(attempt.id)  # pyright: ignore[reportFunctionMemberAccess]
    return AttemptSubmitResponse(attempt_id=attempt.id, status="grading")


def _serialize_attempt(attempt: AssignmentAttempt, db: Session) -> AttemptResponse:
    if attempt.status != "graded":
        return AttemptResponse(status=attempt.status, error=attempt.error)
    answers = db.scalars(
        select(AssignmentAnswer).where(AssignmentAnswer.attempt_id == attempt.id)
    ).all()
    concept_totals: dict[str, list[int]] = {}
    for a in answers:
        totals = concept_totals.setdefault(a.concept_tag, [0, 0])
        totals[1] += 1
        if a.is_correct:
            totals[0] += 1
    return AttemptResponse(
        status="graded",
        overall_score=attempt.overall_score,
        answers=[
            AnswerResult(question_id=a.question_id, is_correct=bool(a.is_correct), feedback=a.feedback or "")
            for a in answers
        ],
        concept_scores=[
            ConceptScore(concept_tag=tag, correct=c, total=t) for tag, (c, t) in concept_totals.items()
        ],
    )


@router.get("/{slug}/assignments/{assignment_id}/attempts/{attempt_id}", response_model=AttemptResponse)
def get_assignment_attempt(slug: str, assignment_id: int, attempt_id: int,
                            db: Session = Depends(get_session), user=Depends(get_current_user)):
    course = db.scalar(select(Course).where(Course.topic_slug == slug))
    if not course:
        raise HTTPException(status_code=404, detail="course not found")
    assignment = db.scalar(
        select(Assignment).where(Assignment.id == assignment_id, Assignment.scope == "global")
    )
    if assignment is None or not _assignment_belongs_to_course(db, assignment, course.id):
        raise HTTPException(status_code=404, detail="assignment not found")
    attempt = db.get(AssignmentAttempt, attempt_id)
    if attempt is None or attempt.assignment_id != assignment.id or attempt.user_id != user.id:
        raise HTTPException(status_code=404, detail="attempt not found")
    return _serialize_attempt(attempt, db)
