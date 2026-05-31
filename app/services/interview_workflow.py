from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.config import get_settings
from app.database import create_session, get_session, update_session_job
from app.schemas import (
    AnswerSet,
    GenerateQuestionsRequest,
    JDAnalysis,
    OutputLanguage,
    QuestionSet,
    ResumeMatch,
    ResumeMatchRequest,
    RoleType,
    SessionResponse,
)
from app.services.ai_service import AIInterviewService
from app.services.mock_service import MockAIInterviewService
from app.services.pdf_parser import extract_resume_text
from app.services.rag_service import RAGQuestionBank


logger = logging.getLogger(__name__)

# Progress checkpoints shown while the frontend polls the background job.
WORKFLOW_STEPS = {
    "parse_resume": 15,
    "analyze_jd": 32,
    "match_resume": 49,
    "generate_questions": 66,
    "generate_answers": 84,
    "save_session": 96,
}


def get_ai_service(demo_mode: bool = False) -> AIInterviewService | MockAIInterviewService:
    # Use deterministic mock data when demo mode is on or OpenAI is not configured.
    settings = get_settings()
    if demo_mode or not settings.openai_api_key:
        return MockAIInterviewService()
    return AIInterviewService(settings)


def effective_demo_mode(demo_mode: bool = False) -> bool:
    return demo_mode or not bool(get_settings().openai_api_key)


def error_detail(message: str, action: str, code: str) -> dict[str, str]:
    return {"message": message, "action": action, "code": code}


def pdf_error_detail(exc: ValueError) -> dict[str, str]:
    return error_detail(
        message=str(exc),
        action="Upload a text-based PDF resume. Scanned image PDFs may need OCR first.",
        code="pdf_parse_error",
    )


def ai_error_detail(exc: Exception) -> tuple[int, dict[str, str]]:
    # Convert common OpenAI/runtime failures into frontend-friendly API errors.
    raw = str(exc)
    lowered = raw.lower()

    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return (
            429,
            error_detail(
                message="Your OpenAI API quota is unavailable or exhausted.",
                action="Check API billing/usage, switch to a cheaper model, or enable Demo mode to preview the app without API calls.",
                code="insufficient_quota",
            ),
        )

    if "api key" in lowered or "missing credentials" in lowered:
        return (
            401,
            error_detail(
                message="OpenAI API key is missing or invalid.",
                action="Add a valid OPENAI_API_KEY to .env, restart the backend, or enable Demo mode.",
                code="api_key_error",
            ),
        )

    if "rate limit" in lowered or "429" in lowered:
        return (
            429,
            error_detail(
                message="OpenAI rate limit reached.",
                action="Wait a moment and retry, use a lower-cost model, or enable Demo mode.",
                code="rate_limit",
            ),
        )

    return (
        502,
        error_detail(
            message="The AI workflow failed before completion.",
            action="Retry once. If it keeps failing, enable Demo mode or check the backend logs.",
            code="ai_workflow_error",
        ),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_events(ai: Any) -> list[dict[str, Any]]:
    if hasattr(ai, "usage_snapshot"):
        return ai.usage_snapshot()
    return []


def _usage_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    # Collapse per-call model usage into totals for job progress reporting.
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


def _persist_or_preview_session(
    *,
    effective_demo_mode_value: bool,
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
    # Demo sessions are returned to the UI but not saved to history.
    if effective_demo_mode_value:
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
        demo_mode=effective_demo_mode_value,
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


def run_session_job(
    *,
    job_id: str,
    user_id: int,
    resume_pdf_bytes: bytes,
    job_description: str,
    role_type: RoleType,
    output_language: OutputLanguage,
    demo_mode: bool,
) -> None:
    # Long-running workflow launched by FastAPI BackgroundTasks.
    steps: list[dict[str, Any]] = []
    all_usage_events: list[dict[str, Any]] = []
    ai: AIInterviewService | MockAIInterviewService | None = None

    def run_step(name: str, operation):
        # Run one workflow step, record latency/usage, and persist progress.
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

    job_start = perf_counter()
    logger.info("job_started", extra={"job_id": job_id, "user_id": user_id, "role_type": role_type, "demo_mode": demo_mode})

    try:
        effective_demo_mode_value = effective_demo_mode(demo_mode)
        update_session_job(job_id, status="running", progress_percent=1)

        resume_text = run_step("parse_resume", lambda: extract_resume_text(resume_pdf_bytes))

        # analyze_jd and match_resume are independent — run them in parallel
        ai_jd = get_ai_service(effective_demo_mode_value)
        ai_match = get_ai_service(effective_demo_mode_value)

        analyze_step = {"name": "analyze_jd", "status": "running", "started_at": _now_iso(), "usage": {}}
        match_step = {"name": "match_resume", "status": "running", "started_at": _now_iso(), "usage": {}}
        steps.extend([analyze_step, match_step])
        update_session_job(
            job_id,
            status="running",
            current_step="analyze_jd",
            progress_percent=max(1, WORKFLOW_STEPS["analyze_jd"] - 10),
            steps=steps,
            usage=_usage_summary(all_usage_events),
        )

        def _do_analyze_jd():
            t = perf_counter()
            result = ai_jd.analyze_jd(job_description, role_type, output_language)
            return result, int((perf_counter() - t) * 1000)

        def _do_match_resume():
            t = perf_counter()
            result = ai_match.match_resume(
                ResumeMatchRequest(
                    resume_text=resume_text,
                    job_description=job_description,
                    role_type=role_type,
                    output_language=output_language,
                    demo_mode=effective_demo_mode_value,
                )
            )
            return result, int((perf_counter() - t) * 1000)

        with ThreadPoolExecutor(max_workers=2) as executor:
            jd_future = executor.submit(_do_analyze_jd)
            match_future = executor.submit(_do_match_resume)

        try:
            jd_analysis, jd_latency = jd_future.result(timeout=120)
            jd_exc = None
        except Exception as e:
            jd_exc = e
            jd_analysis, jd_latency = None, 0

        try:
            resume_match, match_latency = match_future.result(timeout=120)
            match_exc = None
        except Exception as e:
            match_exc = e
            resume_match, match_latency = None, 0

        if jd_exc or match_exc:
            analyze_step.update({
                "status": "failed" if jd_exc else "succeeded",
                "completed_at": _now_iso(),
                **({} if not jd_exc else {"error_message": str(jd_exc)}),
            })
            match_step.update({
                "status": "failed" if match_exc else "succeeded",
                "completed_at": _now_iso(),
                **({} if not match_exc else {"error_message": str(match_exc)}),
            })
            raise (jd_exc or match_exc)

        jd_usage = _usage_events(ai_jd)
        match_usage = _usage_events(ai_match)
        all_usage_events.extend(jd_usage)
        all_usage_events.extend(match_usage)

        analyze_step.update({
            "status": "succeeded",
            "completed_at": _now_iso(),
            "latency_ms": jd_latency,
            "usage": _usage_summary(jd_usage),
        })
        match_step.update({
            "status": "succeeded",
            "completed_at": _now_iso(),
            "latency_ms": match_latency,
            "usage": _usage_summary(match_usage),
        })
        update_session_job(
            job_id,
            current_step="match_resume",
            progress_percent=WORKFLOW_STEPS["match_resume"],
            steps=steps,
            usage=_usage_summary(all_usage_events),
        )

        ai = get_ai_service(effective_demo_mode_value)

        # Retrieve few-shot examples from the semantic question bank.
        rag: RAGQuestionBank | None = None
        few_shot_examples: list[dict] = []
        if not effective_demo_mode_value and get_settings().openai_api_key:
            from openai import OpenAI
            rag = RAGQuestionBank(OpenAI(api_key=get_settings().openai_api_key))
            skills = [s.name for s in jd_analysis.required_technical_skills[:10]]
            few_shot_examples = rag.retrieve_similar(role_type, skills)
            if few_shot_examples:
                logger.info(
                    "rag_retrieved",
                    extra={"job_id": job_id, "count": len(few_shot_examples), "role_type": role_type},
                )

        questions = run_step(
            "generate_questions",
            lambda: ai.generate_questions(
                GenerateQuestionsRequest(
                    resume_text=resume_text,
                    job_description=job_description,
                    role_type=role_type,
                    output_language=output_language,
                    demo_mode=effective_demo_mode_value,
                    jd_analysis=jd_analysis,
                    resume_match=resume_match,
                ),
                few_shot_examples=few_shot_examples or None,
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
                effective_demo_mode_value=effective_demo_mode_value,
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
        logger.info(
            "job_succeeded",
            extra={
                "job_id": job_id,
                "user_id": user_id,
                "latency_ms": int((perf_counter() - job_start) * 1000),
                "total_tokens": _usage_summary(all_usage_events).get("total_tokens"),
            },
        )

        # Store generated questions in the RAG bank for future few-shot retrieval.
        if rag is not None and session.id > 0:
            all_questions = (
                questions.technical_questions
                + questions.project_deep_dive_questions
                + questions.system_design_questions
                + questions.behavioral_questions
            )
            stored = rag.store(session.id, all_questions, session.answers.answers, role_type)
            if stored:
                logger.info(
                    "rag_stored",
                    extra={"job_id": job_id, "session_id": session.id, "count": stored},
                )

    except ValueError as exc:
        detail = pdf_error_detail(exc)
        update_session_job(
            job_id,
            status="failed",
            progress_percent=WORKFLOW_STEPS.get(steps[-1]["name"], 0) if steps else 0,
            error=detail,
            steps=_mark_failed_step(steps, str(exc)),
            usage=_usage_summary(all_usage_events),
            completed=True,
        )
        logger.warning("job_failed", extra={"job_id": job_id, "user_id": user_id, "error_code": detail["code"]})
    except Exception as exc:
        _, detail = ai_error_detail(exc)
        update_session_job(
            job_id,
            status="failed",
            progress_percent=WORKFLOW_STEPS.get(steps[-1]["name"], 0) if steps else 0,
            error=detail,
            steps=_mark_failed_step(steps, detail["message"]),
            usage=_usage_summary(all_usage_events),
            completed=True,
        )
        logger.warning("job_failed", extra={"job_id": job_id, "user_id": user_id, "error_code": detail["code"]})
