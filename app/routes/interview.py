from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.database import create_session, get_session, list_sessions
from app.schemas import (
    AnalyzeJDRequest,
    AnswerResult,
    GenerateAnswerRequest,
    GenerateQuestionsRequest,
    JDAnalysis,
    OutputLanguage,
    QuestionSet,
    ResumeMatch,
    ResumeMatchRequest,
    RoleType,
    SessionResponse,
    SessionSummary,
)
from app.services.ai_service import AIInterviewService
from app.services.mock_service import MockAIInterviewService
from app.services.pdf_parser import extract_resume_text


router = APIRouter(tags=["interview"])


def _ai_service(demo_mode: bool = False) -> AIInterviewService | MockAIInterviewService:
    settings = get_settings()
    if demo_mode or not settings.openai_api_key:
        return MockAIInterviewService()
    return AIInterviewService(settings)


def _effective_demo_mode(demo_mode: bool = False) -> bool:
    return demo_mode or not bool(get_settings().openai_api_key)


def _error_detail(message: str, action: str, code: str) -> dict[str, str]:
    return {"message": message, "action": action, "code": code}


def _raise_pdf_error(exc: ValueError) -> None:
    raise HTTPException(
        status_code=400,
        detail=_error_detail(
            message=str(exc),
            action="Upload a text-based PDF resume. Scanned image PDFs may need OCR first.",
            code="pdf_parse_error",
        ),
    ) from exc


def _raise_ai_error(exc: Exception) -> None:
    raw = str(exc)
    lowered = raw.lower()

    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        detail = _error_detail(
            message="Your OpenAI API quota is unavailable or exhausted.",
            action="Check API billing/usage, switch to a cheaper model, or enable Demo mode to preview the app without API calls.",
            code="insufficient_quota",
        )
        raise HTTPException(status_code=429, detail=detail) from exc

    if "api key" in lowered or "missing credentials" in lowered:
        detail = _error_detail(
            message="OpenAI API key is missing or invalid.",
            action="Add a valid OPENAI_API_KEY to .env, restart the backend, or enable Demo mode.",
            code="api_key_error",
        )
        raise HTTPException(status_code=401, detail=detail) from exc

    if "rate limit" in lowered or "429" in lowered:
        detail = _error_detail(
            message="OpenAI rate limit reached.",
            action="Wait a moment and retry, use a lower-cost model, or enable Demo mode.",
            code="rate_limit",
        )
        raise HTTPException(status_code=429, detail=detail) from exc

    detail = _error_detail(
        message="The AI workflow failed before completion.",
        action="Retry once. If it keeps failing, enable Demo mode or check the backend logs.",
        code="ai_workflow_error",
    )
    raise HTTPException(status_code=502, detail=detail) from exc


@router.post("/analyze-jd", response_model=JDAnalysis)
def analyze_jd(request: AnalyzeJDRequest) -> JDAnalysis:
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
async def match_resume(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
    role_type: RoleType = Form(...),
    output_language: OutputLanguage = Form("Match job description language"),
    demo_mode: bool = Form(False),
) -> ResumeMatch:
    try:
        effective_demo_mode = _effective_demo_mode(demo_mode)
        resume_text = extract_resume_text(await resume_pdf.read())
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
def generate_questions(request: GenerateQuestionsRequest) -> QuestionSet:
    try:
        effective_demo_mode = _effective_demo_mode(request.demo_mode)
        request.demo_mode = effective_demo_mode
        return _ai_service(effective_demo_mode).generate_questions(request)
    except Exception as exc:
        _raise_ai_error(exc)


@router.post("/generate-answer", response_model=AnswerResult)
def generate_answer(request: GenerateAnswerRequest) -> AnswerResult:
    try:
        effective_demo_mode = _effective_demo_mode(request.demo_mode)
        request.demo_mode = effective_demo_mode
        return _ai_service(effective_demo_mode).generate_answer(request)
    except Exception as exc:
        _raise_ai_error(exc)


@router.post("/sessions/from-upload", response_model=SessionResponse)
async def create_session_from_upload(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
    role_type: RoleType = Form(...),
    output_language: OutputLanguage = Form("Match job description language"),
    demo_mode: bool = Form(False),
) -> SessionResponse:
    try:
        effective_demo_mode = _effective_demo_mode(demo_mode)
        resume_text = extract_resume_text(await resume_pdf.read())
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
            role_type=role_type,
            output_language=output_language,
            questions=questions,
        )
        if effective_demo_mode:
            return SessionResponse(
                id=0,
                created_at=datetime.now(timezone.utc).isoformat(),
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
            output_language=output_language,
            demo_mode=effective_demo_mode,
            job_description=job_description,
            resume_text=resume_text,
            jd_analysis=jd_analysis,
            resume_match=resume_match,
            questions=questions,
            answers=answers,
        )
        session = get_session(session_id)
        if session is None:
            raise RuntimeError("Session was saved but could not be loaded.")
        return session
    except ValueError as exc:
        _raise_pdf_error(exc)
    except Exception as exc:
        _raise_ai_error(exc)


@router.get("/sessions", response_model=list[SessionSummary])
def sessions() -> list[SessionSummary]:
    return list_sessions()


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def session_detail(session_id: int) -> SessionResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session
