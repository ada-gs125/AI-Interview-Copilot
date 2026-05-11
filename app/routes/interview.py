from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from app.config import get_settings
from app.database import (
    create_session,
    create_session_job,
    delete_expired_sessions,
    delete_session,
    get_session,
    get_session_job,
    list_sessions,
    update_session_job,
)
from app.dependencies import current_user
from app.schemas import (
    AnalyzeJDRequest,
    AnswerResult,
    AnswerSet,
    GenerateAnswerRequest,
    GenerateQuestionsRequest,
    JDAnalysis,
    OutputLanguage,
    QuestionSet,
    ResumeMatch,
    ResumeMatchRequest,
    RoleType,
    SessionJobCreateResponse,
    SessionJobResponse,
    SessionResponse,
    SessionSummary,
    UserResponse,
)
from app.services.ai_service import AIInterviewService
from app.services.mock_service import MockAIInterviewService
from app.services.pdf_parser import extract_resume_text


router = APIRouter(tags=["interview"])

WORKFLOW_STEPS = {
    "parse_resume": 15,
    "analyze_jd": 32,
    "match_resume": 49,
    "generate_questions": 66,
    "generate_answers": 84,
    "save_session": 96,
}


def _ai_service(demo_mode: bool = False) -> AIInterviewService | MockAIInterviewService:
    settings = get_settings()
    if demo_mode or not settings.openai_api_key:
        return MockAIInterviewService()
    return AIInterviewService(settings)


def _effective_demo_mode(demo_mode: bool = False) -> bool:
    return demo_mode or not bool(get_settings().openai_api_key)


def _error_detail(message: str, action: str, code: str) -> dict[str, str]:
    return {"message": message, "action": action, "code": code}


def _pdf_error_detail(exc: ValueError) -> dict[str, str]:
    return _error_detail(
        message=str(exc),
        action="Upload a text-based PDF resume. Scanned image PDFs may need OCR first.",
        code="pdf_parse_error",
    )


def _raise_pdf_error(exc: ValueError) -> None:
    raise HTTPException(status_code=400, detail=_pdf_error_detail(exc)) from exc


def _ai_error_detail(exc: Exception) -> tuple[int, dict[str, str]]:
    raw = str(exc)
    lowered = raw.lower()

    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return (
            429,
            _error_detail(
                message="Your OpenAI API quota is unavailable or exhausted.",
                action="Check API billing/usage, switch to a cheaper model, or enable Demo mode to preview the app without API calls.",
                code="insufficient_quota",
            ),
        )

    if "api key" in lowered or "missing credentials" in lowered:
        return (
            401,
            _error_detail(
                message="OpenAI API key is missing or invalid.",
                action="Add a valid OPENAI_API_KEY to .env, restart the backend, or enable Demo mode.",
                code="api_key_error",
            ),
        )

    if "rate limit" in lowered or "429" in lowered:
        return (
            429,
            _error_detail(
                message="OpenAI rate limit reached.",
                action="Wait a moment and retry, use a lower-cost model, or enable Demo mode.",
                code="rate_limit",
            ),
        )

    return (
        502,
        _error_detail(
            message="The AI workflow failed before completion.",
            action="Retry once. If it keeps failing, enable Demo mode or check the backend logs.",
            code="ai_workflow_error",
        ),
    )


def _raise_ai_error(exc: Exception) -> None:
    status_code, detail = _ai_error_detail(exc)
    raise HTTPException(status_code=status_code, detail=detail) from exc


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_events(ai: Any) -> list[dict[str, Any]]:
    if hasattr(ai, "usage_snapshot"):
        return ai.usage_snapshot()
    return []


def _usage_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"calls": events, "call_count": len(events)}
    for key in ("input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"):
        values = [
            event.get(key)
            for event in events
            if isinstance(event.get(key), int) or isinstance(event.get(key), float)
        ]
        if values:
            summary[key] = round(sum(values), 6) if key == "estimated_cost_usd" else sum(values)
    return summary


def _mark_failed_step(steps: list[dict[str, Any]], message: str) -> list[dict[str, Any]]:
    if steps and steps[-1]["status"] == "running":
        steps[-1].update(
            {
                "status": "failed",
                "completed_at": _now_iso(),
                "error_message": message,
            }
        )
    return steps


def _run_session_workflow(
    *,
    user_id: int,
    resume_pdf_bytes: bytes,
    job_description: str,
    role_type: RoleType,
    output_language: OutputLanguage,
    demo_mode: bool,
) -> SessionResponse:
    effective_demo_mode = _effective_demo_mode(demo_mode)
    resume_text = extract_resume_text(resume_pdf_bytes)
    ai = _ai_service(effective_demo_mode)
    jd_analysis = ai.analyze_jd(job_description, role_type, output_language)
    resume_match = ai.match_resume(
        ResumeMatchRequest(
            resume_text=resume_text,
            job_description=job_description,
            role_type=role_type,
            output_language=output_language,
            demo_mode=effective_demo_mode,
        )
    )
    questions = ai.generate_questions(
        GenerateQuestionsRequest(
            resume_text=resume_text,
            job_description=job_description,
            role_type=role_type,
            output_language=output_language,
            demo_mode=effective_demo_mode,
            jd_analysis=jd_analysis,
            resume_match=resume_match,
        )
    )
    answers = ai.generate_answers_for_question_set(
        resume_text=resume_text,
        job_description=job_description,
        role_type=role_type,
        output_language=output_language,
        questions=questions,
    )
    return _persist_or_preview_session(
        effective_demo_mode=effective_demo_mode,
        user_id=user_id,
        role_type=role_type,
        output_language=output_language,
        job_description=job_description,
        resume_text=resume_text,
        jd_analysis=jd_analysis,
        resume_match=resume_match,
        questions=questions,
        answers=answers,
    )


def _persist_or_preview_session(
    *,
    effective_demo_mode: bool,
    user_id: int,
    role_type: RoleType,
    output_language: OutputLanguage,
    job_description: str,
    resume_text: str,
    jd_analysis: JDAnalysis,
    resume_match: ResumeMatch,
    questions: QuestionSet,
    answers: AnswerSet,
) -> SessionResponse:
    if effective_demo_mode:
        return SessionResponse(
            id=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            role_type=role_type,
            output_language=output_language,
            demo_mode=True,
            job_description=job_description,
            resume_text=resume_text,
            jd_analysis=jd_analysis,
            resume_match=resume_match,
            questions=questions,
            answers=answers,
        )

    session_id = create_session(
        role_type=role_type,
        user_id=user_id,
        output_language=output_language,
        demo_mode=effective_demo_mode,
        job_description=job_description,
        resume_text=resume_text,
        jd_analysis=jd_analysis,
        resume_match=resume_match,
        questions=questions,
        answers=answers,
    )
    session = get_session(session_id, user_id=user_id)
    if session is None:
        raise RuntimeError("Session was saved but could not be loaded.")
    return session


def _run_session_job(
    *,
    job_id: str,
    user_id: int,
    resume_pdf_bytes: bytes,
    job_description: str,
    role_type: RoleType,
    output_language: OutputLanguage,
    demo_mode: bool,
) -> None:
    steps: list[dict[str, Any]] = []
    all_usage_events: list[dict[str, Any]] = []
    ai: AIInterviewService | MockAIInterviewService | None = None

    def run_step(name: str, operation):
        nonlocal all_usage_events
        step = {"name": name, "status": "running", "started_at": _now_iso(), "usage": {}}
        steps.append(step)
        update_session_job(
            job_id,
            status="running",
            current_step=name,
            progress_percent=max(1, WORKFLOW_STEPS[name] - 10),
            steps=steps,
            usage=_usage_summary(all_usage_events),
        )

        usage_before = len(_usage_events(ai))
        start = perf_counter()
        result = operation()
        latency_ms = int((perf_counter() - start) * 1000)
        usage_after = _usage_events(ai)
        step_usage_events = usage_after[usage_before:]
        all_usage_events.extend(step_usage_events)

        step.update(
            {
                "status": "succeeded",
                "completed_at": _now_iso(),
                "latency_ms": latency_ms,
                "usage": _usage_summary(step_usage_events),
            }
        )
        update_session_job(
            job_id,
            current_step=name,
            progress_percent=WORKFLOW_STEPS[name],
            steps=steps,
            usage=_usage_summary(all_usage_events),
        )
        return result

    try:
        effective_demo_mode = _effective_demo_mode(demo_mode)
        update_session_job(job_id, status="running", progress_percent=1)

        resume_text = run_step("parse_resume", lambda: extract_resume_text(resume_pdf_bytes))
        ai = _ai_service(effective_demo_mode)
        jd_analysis = run_step(
            "analyze_jd",
            lambda: ai.analyze_jd(job_description, role_type, output_language),
        )
        resume_match = run_step(
            "match_resume",
            lambda: ai.match_resume(
                ResumeMatchRequest(
                    resume_text=resume_text,
                    job_description=job_description,
                    role_type=role_type,
                    output_language=output_language,
                    demo_mode=effective_demo_mode,
                )
            ),
        )
        questions = run_step(
            "generate_questions",
            lambda: ai.generate_questions(
                GenerateQuestionsRequest(
                    resume_text=resume_text,
                    job_description=job_description,
                    role_type=role_type,
                    output_language=output_language,
                    demo_mode=effective_demo_mode,
                    jd_analysis=jd_analysis,
                    resume_match=resume_match,
                )
            ),
        )
        answers = run_step(
            "generate_answers",
            lambda: ai.generate_answers_for_question_set(
                resume_text=resume_text,
                job_description=job_description,
                role_type=role_type,
                output_language=output_language,
                questions=questions,
            ),
        )
        session = run_step(
            "save_session",
            lambda: _persist_or_preview_session(
                effective_demo_mode=effective_demo_mode,
                user_id=user_id,
                role_type=role_type,
                output_language=output_language,
                job_description=job_description,
                resume_text=resume_text,
                jd_analysis=jd_analysis,
                resume_match=resume_match,
                questions=questions,
                answers=answers,
            ),
        )
        update_session_job(
            job_id,
            status="succeeded",
            current_step="completed",
            progress_percent=100,
            session_id=session.id if session.id else None,
            steps=steps,
            usage=_usage_summary(all_usage_events),
            result=session,
            completed=True,
        )
    except ValueError as exc:
        detail = _pdf_error_detail(exc)
        update_session_job(
            job_id,
            status="failed",
            progress_percent=WORKFLOW_STEPS.get(steps[-1]["name"], 0) if steps else 0,
            error=detail,
            steps=_mark_failed_step(steps, str(exc)),
            usage=_usage_summary(all_usage_events),
            completed=True,
        )
    except Exception as exc:
        _, detail = _ai_error_detail(exc)
        update_session_job(
            job_id,
            status="failed",
            progress_percent=WORKFLOW_STEPS.get(steps[-1]["name"], 0) if steps else 0,
            error=detail,
            steps=_mark_failed_step(steps, detail["message"]),
            usage=_usage_summary(all_usage_events),
            completed=True,
        )


@router.post("/analyze-jd", response_model=JDAnalysis)
def analyze_jd(request: AnalyzeJDRequest, user: UserResponse = Depends(current_user)) -> JDAnalysis:
    try:
        effective_demo_mode = _effective_demo_mode(request.demo_mode)
        return _ai_service(effective_demo_mode).analyze_jd(
            request.job_description,
            request.role_type,
            request.output_language,
        )
    except Exception as exc:
        _raise_ai_error(exc)


@router.post("/match-resume", response_model=ResumeMatch)
def match_resume(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
    role_type: RoleType = Form(...),
    output_language: OutputLanguage = Form("Match job description language"),
    demo_mode: bool = Form(False),
    user: UserResponse = Depends(current_user),
) -> ResumeMatch:
    try:
        effective_demo_mode = _effective_demo_mode(demo_mode)
        resume_text = extract_resume_text(resume_pdf.file.read())
        return _ai_service(effective_demo_mode).match_resume(
            ResumeMatchRequest(
                resume_text=resume_text,
                job_description=job_description,
                role_type=role_type,
                output_language=output_language,
                demo_mode=effective_demo_mode,
            )
        )
    except ValueError as exc:
        _raise_pdf_error(exc)
    except Exception as exc:
        _raise_ai_error(exc)


@router.post("/generate-questions", response_model=QuestionSet)
def generate_questions(request: GenerateQuestionsRequest, user: UserResponse = Depends(current_user)) -> QuestionSet:
    try:
        effective_demo_mode = _effective_demo_mode(request.demo_mode)
        request.demo_mode = effective_demo_mode
        return _ai_service(effective_demo_mode).generate_questions(request)
    except Exception as exc:
        _raise_ai_error(exc)


@router.post("/generate-answer", response_model=AnswerResult)
def generate_answer(request: GenerateAnswerRequest, user: UserResponse = Depends(current_user)) -> AnswerResult:
    try:
        effective_demo_mode = _effective_demo_mode(request.demo_mode)
        request.demo_mode = effective_demo_mode
        return _ai_service(effective_demo_mode).generate_answer(request)
    except Exception as exc:
        _raise_ai_error(exc)


@router.post("/sessions/from-upload", response_model=SessionResponse)
def create_session_from_upload(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
    role_type: RoleType = Form(...),
    output_language: OutputLanguage = Form("Match job description language"),
    demo_mode: bool = Form(False),
    user: UserResponse = Depends(current_user),
) -> SessionResponse:
    try:
        return _run_session_workflow(
            user_id=user.id,
            resume_pdf_bytes=resume_pdf.file.read(),
            job_description=job_description,
            role_type=role_type,
            output_language=output_language,
            demo_mode=demo_mode,
        )
    except ValueError as exc:
        _raise_pdf_error(exc)
    except Exception as exc:
        _raise_ai_error(exc)


@router.post(
    "/sessions/jobs",
    response_model=SessionJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_session_job_from_upload(
    background_tasks: BackgroundTasks,
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
    role_type: RoleType = Form(...),
    output_language: OutputLanguage = Form("Match job description language"),
    demo_mode: bool = Form(False),
    user: UserResponse = Depends(current_user),
) -> SessionJobCreateResponse:
    job_id = str(uuid4())
    effective_demo_mode = _effective_demo_mode(demo_mode)
    create_session_job(
        job_id=job_id,
        user_id=user.id,
        role_type=role_type,
        output_language=output_language,
        demo_mode=effective_demo_mode,
    )
    background_tasks.add_task(
        _run_session_job,
        job_id=job_id,
        user_id=user.id,
        resume_pdf_bytes=await resume_pdf.read(),
        job_description=job_description,
        role_type=role_type,
        output_language=output_language,
        demo_mode=effective_demo_mode,
    )
    return SessionJobCreateResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/sessions/jobs/{job_id}",
    )


@router.get("/sessions", response_model=list[SessionSummary])
def sessions(user: UserResponse = Depends(current_user)) -> list[SessionSummary]:
    delete_expired_sessions(user_id=user.id, retention_days=get_settings().session_retention_days)
    return list_sessions(user_id=user.id)


@router.get("/sessions/jobs/{job_id}", response_model=SessionJobResponse)
def session_job_detail(job_id: str, user: UserResponse = Depends(current_user)) -> SessionJobResponse:
    job = get_session_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Session job not found.")
    return job


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def session_detail(session_id: int, user: UserResponse = Depends(current_user)) -> SessionResponse:
    session = get_session(session_id, user_id=user.id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session_detail(session_id: int, user: UserResponse = Depends(current_user)) -> None:
    deleted = delete_session(session_id, user_id=user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
