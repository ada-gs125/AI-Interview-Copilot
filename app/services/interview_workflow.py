"""Background workflow orchestration for generating interview prep sessions."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Union

from openai import OpenAI

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
from app.services.pdf_parser import ResumePDFParseError, extract_resume_text
from app.services.rag_service import RAGQuestionBank


logger = logging.getLogger(__name__)

AIService = Union[AIInterviewService, MockAIInterviewService]

# Progress checkpoints shown while the frontend polls the background job.
WORKFLOW_STEPS = {
    "parse_resume": 15,
    "analyze_jd": 32,
    "match_resume": 49,
    "generate_questions": 66,
    "generate_answers": 84,
    "save_session": 96,
}


def get_ai_service(demo_mode: bool = False) -> AIService:
    # Use deterministic mock data when demo mode is on or OpenAI is not configured.
    settings = get_settings()
    if demo_mode or not settings.openai_api_key:
        return MockAIInterviewService()
    return AIInterviewService(settings)


def effective_demo_mode(demo_mode: bool = False) -> bool:
    # Force demo mode when the deployment has no OpenAI credentials.
    return demo_mode or not bool(get_settings().openai_api_key)


def error_detail(message: str, action: str, code: str) -> dict[str, str]:
    return {"message": message, "action": action, "code": code}


def pdf_error_detail(exc: ValueError) -> dict[str, str]:
    # PDF parsing failures are user-fixable, so return a specific action.
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


class InterviewWorkflowRunner:
    """Stateful runner for one background interview-prep job."""

    def __init__(
        self,
        *,
        job_id: str,
        user_id: int,
        resume_pdf_bytes: bytes,
        job_description: str,
        role_type: RoleType,
        output_language: OutputLanguage,
        demo_mode: bool,
    ) -> None:
        self.job_id = job_id
        self.user_id = user_id
        self.resume_pdf_bytes = resume_pdf_bytes
        self.job_description = job_description
        self.role_type = role_type
        self.output_language = output_language
        self.demo_mode = demo_mode

        self.effective_demo_mode = effective_demo_mode(demo_mode)
        self.steps: list[dict[str, Any]] = []
        self.all_usage_events: list[dict[str, Any]] = []
        self.ai: AIService | None = None
        self.rag: RAGQuestionBank | None = None
        self.started_at = perf_counter()

    def run(self) -> None:
        logger.info(
            "job_started",
            extra={
                "job_id": self.job_id,
                "user_id": self.user_id,
                "role_type": self.role_type,
                "demo_mode": self.demo_mode,
            },
        )

        try:
            update_session_job(self.job_id, status="running", progress_percent=1)
            session = self._run_workflow()
            self._complete_job(session)
            self._store_rag_examples(session)
        except ResumePDFParseError as exc:
            self._fail_job(pdf_error_detail(exc), str(exc))
        except Exception as exc:
            _, detail = ai_error_detail(exc)
            self._fail_job(detail, detail["message"])

    def _run_workflow(self) -> SessionResponse:
        resume_text = self._run_step("parse_resume", lambda: extract_resume_text(self.resume_pdf_bytes))
        jd_analysis, resume_match = self._analyze_job_and_resume(resume_text)

        self.ai = get_ai_service(self.effective_demo_mode)
        few_shot_examples = self._retrieve_few_shot_examples(jd_analysis)

        questions = self._run_step(
            "generate_questions",
            lambda: self.ai.generate_questions(
                GenerateQuestionsRequest(
                    resume_text=resume_text,
                    job_description=self.job_description,
                    role_type=self.role_type,
                    output_language=self.output_language,
                    demo_mode=self.effective_demo_mode,
                    jd_analysis=jd_analysis,
                    resume_match=resume_match,
                ),
                few_shot_examples=few_shot_examples or None,
            ),
        )
        answers = self._run_step(
            "generate_answers",
            lambda: self.ai.generate_answers_for_question_set(
                resume_text=resume_text,
                job_description=self.job_description,
                role_type=self.role_type,
                output_language=self.output_language,
                questions=questions,
            ),
        )
        return self._run_step(
            "save_session",
            lambda: self._persist_or_preview_session(
                resume_text=resume_text,
                jd_analysis=jd_analysis,
                resume_match=resume_match,
                questions=questions,
                answers=answers,
            ),
        )

    def _run_step(self, name: str, operation: Callable[[], Any]):
        # Run one workflow step, record latency/usage, and persist progress.
        step = {"name": name, "status": "running", "started_at": _now_iso(), "usage": {}}
        self.steps.append(step)
        update_session_job(
            self.job_id,
            status="running",
            current_step=name,
            progress_percent=max(1, WORKFLOW_STEPS[name] - 10),
            steps=self.steps,
            usage=_usage_summary(self.all_usage_events),
        )

        usage_before = len(_usage_events(self.ai))
        start = perf_counter()
        result = operation()
        latency_ms = int((perf_counter() - start) * 1000)
        usage_after = _usage_events(self.ai)
        step_usage_events = usage_after[usage_before:]
        self.all_usage_events.extend(step_usage_events)

        step.update(
            {
                "status": "succeeded",
                "completed_at": _now_iso(),
                "latency_ms": latency_ms,
                "usage": _usage_summary(step_usage_events),
            }
        )
        update_session_job(
            self.job_id,
            current_step=name,
            progress_percent=WORKFLOW_STEPS[name],
            steps=self.steps,
            usage=_usage_summary(self.all_usage_events),
        )
        return result

    def _analyze_job_and_resume(self, resume_text: str) -> tuple[JDAnalysis, ResumeMatch]:
        ai_jd = get_ai_service(self.effective_demo_mode)
        ai_match = get_ai_service(self.effective_demo_mode)
        analyze_step = {"name": "analyze_jd", "status": "running", "started_at": _now_iso(), "usage": {}}
        match_step = {"name": "match_resume", "status": "running", "started_at": _now_iso(), "usage": {}}
        self.steps.extend([analyze_step, match_step])
        update_session_job(
            self.job_id,
            status="running",
            current_step="analyze_jd",
            progress_percent=max(1, WORKFLOW_STEPS["analyze_jd"] - 10),
            steps=self.steps,
            usage=_usage_summary(self.all_usage_events),
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            jd_future = executor.submit(self._do_analyze_jd, ai_jd)
            match_future = executor.submit(self._do_match_resume, ai_match, resume_text)

        jd_analysis, jd_latency, jd_exc = self._resolve_future(jd_future)
        resume_match, match_latency, match_exc = self._resolve_future(match_future)

        if jd_exc or match_exc:
            self._finish_parallel_step(analyze_step, jd_latency, _usage_events(ai_jd), jd_exc)
            self._finish_parallel_step(match_step, match_latency, _usage_events(ai_match), match_exc)
            exc = jd_exc or match_exc
            raise exc

        jd_usage = _usage_events(ai_jd)
        match_usage = _usage_events(ai_match)
        self.all_usage_events.extend(jd_usage)
        self.all_usage_events.extend(match_usage)

        self._finish_parallel_step(analyze_step, jd_latency, jd_usage, None)
        self._finish_parallel_step(match_step, match_latency, match_usage, None)
        update_session_job(
            self.job_id,
            current_step="match_resume",
            progress_percent=WORKFLOW_STEPS["match_resume"],
            steps=self.steps,
            usage=_usage_summary(self.all_usage_events),
        )
        return jd_analysis, resume_match

    def _do_analyze_jd(self, ai_jd: AIService) -> tuple[JDAnalysis, int]:
        start = perf_counter()
        result = ai_jd.analyze_jd(self.job_description, self.role_type, self.output_language)
        return result, int((perf_counter() - start) * 1000)

    def _do_match_resume(self, ai_match: AIService, resume_text: str) -> tuple[ResumeMatch, int]:
        start = perf_counter()
        result = ai_match.match_resume(
            ResumeMatchRequest(
                resume_text=resume_text,
                job_description=self.job_description,
                role_type=self.role_type,
                output_language=self.output_language,
                demo_mode=self.effective_demo_mode,
            )
        )
        return result, int((perf_counter() - start) * 1000)

    def _resolve_future(self, future) -> tuple[Any, int, Exception | None]:
        try:
            result, latency_ms = future.result(timeout=120)
            return result, latency_ms, None
        except Exception as exc:
            return None, 0, exc

    def _finish_parallel_step(
        self,
        step: dict[str, Any],
        latency_ms: int,
        usage_events: list[dict[str, Any]],
        error: Exception | None,
    ) -> None:
        step.update(
            {
                "status": "failed" if error else "succeeded",
                "completed_at": _now_iso(),
                "latency_ms": latency_ms,
                "usage": _usage_summary(usage_events),
            }
        )
        if error is not None:
            step["error_message"] = str(error)

    def _retrieve_few_shot_examples(self, jd_analysis: JDAnalysis) -> list[dict[str, Any]]:
        # Retrieve few-shot examples from the semantic question bank.
        if self.effective_demo_mode or not get_settings().openai_api_key:
            return []

        # Reuse the existing OpenAI client rather than creating a second HTTP connection pool.
        client = self.ai.client if isinstance(self.ai, AIInterviewService) else OpenAI(api_key=get_settings().openai_api_key)
        self.rag = RAGQuestionBank(client)
        skills = [s.name for s in jd_analysis.required_technical_skills[:10]]
        few_shot_examples = self.rag.retrieve_similar(self.role_type, skills, user_id=self.user_id)
        if few_shot_examples:
            logger.info(
                "rag_retrieved",
                extra={"job_id": self.job_id, "count": len(few_shot_examples), "role_type": self.role_type},
            )
        return few_shot_examples

    def _persist_or_preview_session(
        self,
        *,
        resume_text: str,
        jd_analysis: JDAnalysis,
        resume_match: ResumeMatch,
        questions: QuestionSet,
        answers: AnswerSet,
    ) -> SessionResponse:
        # Demo sessions are returned to the UI but not saved to history.
        if self.effective_demo_mode:
            return SessionResponse(
                id=0,
                created_at=datetime.now(timezone.utc).isoformat(),
                user_id=self.user_id,
                role_type=self.role_type,
                output_language=self.output_language,
                demo_mode=True,
                job_description=self.job_description,
                resume_text=resume_text,
                jd_analysis=jd_analysis,
                resume_match=resume_match,
                questions=questions,
                answers=answers,
            )

        session_id = create_session(
            role_type=self.role_type,
            user_id=self.user_id,
            output_language=self.output_language,
            demo_mode=self.effective_demo_mode,
            job_description=self.job_description,
            resume_text=resume_text,
            jd_analysis=jd_analysis,
            resume_match=resume_match,
            questions=questions,
            answers=answers,
        )
        session = get_session(session_id, user_id=self.user_id)
        if session is None:
            raise RuntimeError("Session was saved but could not be loaded.")
        return session

    def _complete_job(self, session: SessionResponse) -> None:
        update_session_job(
            self.job_id,
            status="succeeded",
            current_step="completed",
            progress_percent=100,
            session_id=session.id if session.id else None,
            steps=self.steps,
            usage=_usage_summary(self.all_usage_events),
            result=session,
            completed=True,
        )
        logger.info(
            "job_succeeded",
            extra={
                "job_id": self.job_id,
                "user_id": self.user_id,
                "latency_ms": int((perf_counter() - self.started_at) * 1000),
                "total_tokens": _usage_summary(self.all_usage_events).get("total_tokens"),
            },
        )

    def _store_rag_examples(self, session: SessionResponse) -> None:
        # Store generated questions in the RAG bank for future few-shot retrieval.
        if self.effective_demo_mode or self.rag is None or session.id <= 0:
            return

        questions = session.questions
        all_questions = (
            questions.technical_questions
            + questions.project_deep_dive_questions
            + questions.system_design_questions
            + questions.behavioral_questions
        )
        stored = self.rag.store(session.id, all_questions, session.answers.answers, self.role_type, user_id=self.user_id)
        if stored:
            logger.info(
                "rag_stored",
                extra={"job_id": self.job_id, "session_id": session.id, "count": stored},
            )

    def _fail_job(self, detail: dict[str, str], step_message: str) -> None:
        self._mark_failed_step(step_message)
        update_session_job(
            self.job_id,
            status="failed",
            progress_percent=self._current_progress(),
            error=detail,
            steps=self.steps,
            usage=_usage_summary(self.all_usage_events),
            completed=True,
        )
        logger.warning(
            "job_failed",
            extra={"job_id": self.job_id, "user_id": self.user_id, "error_code": detail["code"]},
        )

    def _mark_failed_step(self, message: str) -> None:
        if self.steps and self.steps[-1]["status"] == "running":
            self.steps[-1].update(
                {
                    "status": "failed",
                    "completed_at": _now_iso(),
                    "error_message": message,
                }
            )

    def _current_progress(self) -> int:
        if not self.steps:
            return 0
        return WORKFLOW_STEPS.get(self.steps[-1]["name"], 0)


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
    """Run one interview workflow job from FastAPI BackgroundTasks."""
    InterviewWorkflowRunner(
        job_id=job_id,
        user_id=user_id,
        resume_pdf_bytes=resume_pdf_bytes,
        job_description=job_description,
        role_type=role_type,
        output_language=output_language,
        demo_mode=demo_mode,
    ).run()
