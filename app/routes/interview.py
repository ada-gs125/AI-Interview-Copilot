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
from app.services.pdf_parser import extract_resume_text


router = APIRouter(tags=["interview"])


def _ai_service() -> AIInterviewService:
    return AIInterviewService(get_settings())


@router.post("/analyze-jd", response_model=JDAnalysis)
def analyze_jd(request: AnalyzeJDRequest) -> JDAnalysis:
    try:
        return _ai_service().analyze_jd(
            request.job_description,
            request.role_type,
            request.output_language,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/match-resume", response_model=ResumeMatch)
async def match_resume(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
    role_type: RoleType = Form(...),
    output_language: OutputLanguage = Form("Match job description language"),
) -> ResumeMatch:
    try:
        resume_text = extract_resume_text(await resume_pdf.read())
        return _ai_service().match_resume(
            ResumeMatchRequest(
                resume_text=resume_text,
                job_description=job_description,
                role_type=role_type,
                output_language=output_language,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/generate-questions", response_model=QuestionSet)
def generate_questions(request: GenerateQuestionsRequest) -> QuestionSet:
    try:
        return _ai_service().generate_questions(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/generate-answer", response_model=AnswerResult)
def generate_answer(request: GenerateAnswerRequest) -> AnswerResult:
    try:
        return _ai_service().generate_answer(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/sessions/from-upload", response_model=SessionResponse)
async def create_session_from_upload(
    resume_pdf: UploadFile = File(...),
    job_description: str = Form(...),
    role_type: RoleType = Form(...),
    output_language: OutputLanguage = Form("Match job description language"),
) -> SessionResponse:
    try:
        resume_text = extract_resume_text(await resume_pdf.read())
        ai = _ai_service()
        jd_analysis = ai.analyze_jd(job_description, role_type, output_language)
        resume_match = ai.match_resume(
            ResumeMatchRequest(
                resume_text=resume_text,
                job_description=job_description,
                role_type=role_type,
                output_language=output_language,
            )
        )
        questions = ai.generate_questions(
            GenerateQuestionsRequest(
                resume_text=resume_text,
                job_description=job_description,
                role_type=role_type,
                output_language=output_language,
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
        session_id = create_session(
            role_type=role_type,
            output_language=output_language,
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/sessions", response_model=list[SessionSummary])
def sessions() -> list[SessionSummary]:
    return list_sessions()


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def session_detail(session_id: int) -> SessionResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session
