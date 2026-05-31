from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.database import (
    create_session_job,
    delete_expired_sessions,
    delete_session,
    get_session,
    get_session_job,
    list_sessions,
)
from app.dependencies import current_user
from app.schemas import (
    GenerateAnswerRequest,
    OutputLanguage,
    RoleType,
    SessionJobCreateResponse,
    SessionJobResponse,
    SessionResponse,
    SessionSummary,
    UserResponse,
)
from app.services.interview_workflow import (
    ai_error_detail,
    effective_demo_mode,
    get_ai_service,
    run_session_job,
)


router = APIRouter(tags=["interview"])


@router.post("/generate-answer/stream")
def generate_answer_stream(
    request: GenerateAnswerRequest,
    user: UserResponse = Depends(current_user),
) -> StreamingResponse:
    # Stream regenerated answer tokens to Streamlit as server-sent events.
    effective_demo_mode_value = effective_demo_mode(request.demo_mode)
    ai = get_ai_service(effective_demo_mode_value)

    def event_stream():
        try:
            for chunk in ai.stream_answer(request):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            _, detail = ai_error_detail(exc)
            yield f"event: error\ndata: {json.dumps(detail)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
    # Create a queued job and return immediately; the frontend polls status_url.
    job_id = str(uuid4())
    effective_demo_mode_value = effective_demo_mode(demo_mode)
    create_session_job(
        job_id=job_id,
        user_id=user.id,
        role_type=role_type,
        output_language=output_language,
        demo_mode=effective_demo_mode_value,
    )
    background_tasks.add_task(
        run_session_job,
        job_id=job_id,
        user_id=user.id,
        resume_pdf_bytes=await resume_pdf.read(),
        job_description=job_description,
        role_type=role_type,
        output_language=output_language,
        demo_mode=effective_demo_mode_value,
    )
    return SessionJobCreateResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/sessions/jobs/{job_id}",
    )


@router.get("/sessions", response_model=list[SessionSummary])
def sessions(user: UserResponse = Depends(current_user)) -> list[SessionSummary]:
    # Keep only recent sessions for this user, then return sidebar summaries.
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
