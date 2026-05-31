"""Background workflow orchestration for generating interview prep sessions."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, Literal, Union

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WorkflowStep:
    name: str
    status: Literal["running", "succeeded", "failed"] = "running"
    started_at: str = field(default_factory=_now_iso)
    completed_at: str | None = None
    latency_ms: int | None = None
    usage: dict = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "usage": self.usage,
        }
        if self.completed_at is not None:
            d["completed_at"] = self.completed_at
        if self.latency_ms is not None:
            d["latency_ms"] = self.latency_ms
        if self.error_message is not None:
            d["error_message"] = self.error_message
        return d


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
            if isinstance(event.get(key), (int, float))
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
        self.steps: list[WorkflowStep] = []
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

    # ------------------------------------------------------------------
    # Workflow orchestration
    # ------------------------------------------------------------------

    def _run_workflow(self) -> SessionResponse:
        resume_text = self._run_step("parse_resume", lambda: extract_resume_text(self.resume_pdf_bytes))
        jd_analysis, resume_match = self._analyze_job_and_resume(resume_text)

        self.ai = get_ai_service(self.effective_demo_mode)
        self.rag, few_shot_examples = self._init_rag(jd_analysis)

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

    def _analyze_job_and_resume(self, resume_text: str) -> tuple[JDAnalysis, ResumeMatch]:
        jd_analysis, resume_match = self._run_parallel_steps([
            ("analyze_jd", self._do_analyze_jd),
            ("match_resume", lambda: self._do_match_resume(resume_text)),
        ])
        return jd_analysis, resume_match

    # ------------------------------------------------------------------
    # Step execution primitives
    # ------------------------------------------------------------------

    def _run_step(self, name: str, operation: Callable[[], Any]) -> Any:
        """Run a sequential step: track progress, latency, and usage."""
        step = WorkflowStep(name=name)
        self.steps.append(step)
        update_session_job(
            self.job_id,
            status="running",
            current_step=name,
            progress_percent=max(1, WORKFLOW_STEPS[name] - 10),
            steps=self._steps_as_dicts(),
            usage=_usage_summary(self.all_usage_events),
        )

        usage_before = len(_usage_events(self.ai))
        start = perf_counter()
        result = operation()
        latency_ms = int((perf_counter() - start) * 1000)
        step_usage_events = _usage_events(self.ai)[usage_before:]
        self.all_usage_events.extend(step_usage_events)

        step.status = "succeeded"
        step.completed_at = _now_iso()
        step.latency_ms = latency_ms
        step.usage = _usage_summary(step_usage_events)

        update_session_job(
            self.job_id,
            current_step=name,
            progress_percent=WORKFLOW_STEPS[name],
            steps=self._steps_as_dicts(),
            usage=_usage_summary(self.all_usage_events),
        )
        return result

    def _run_parallel_steps(
        self,
        tasks: list[tuple[str, Callable[[], tuple[Any, int, list[dict[str, Any]]]]]],
    ) -> list[Any]:
        """Run steps concurrently; each callable returns (result, latency_ms, usage_events)."""
        steps = [WorkflowStep(name=name) for name, _ in tasks]
        for step in steps:
            self.steps.append(step)

        update_session_job(
            self.job_id,
            status="running",
            current_step=tasks[0][0],
            progress_percent=max(1, WORKFLOW_STEPS[tasks[0][0]] - 10),
            steps=self._steps_as_dicts(),
            usage=_usage_summary(self.all_usage_events),
        )

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(fn) for _, fn in tasks]

        results: list[Any] = []
        first_exc: Exception | None = None
        for step, future in zip(steps, futures):
            try:
                result, latency_ms, usage_events = future.result(timeout=120)
                step.status = "succeeded"
                step.completed_at = _now_iso()
                step.latency_ms = latency_ms
                step.usage = _usage_summary(usage_events)
                self.all_usage_events.extend(usage_events)
                results.append(result)
            except Exception as exc:
                step.status = "failed"
                step.completed_at = _now_iso()
                step.error_message = str(exc)
                results.append(None)
                if first_exc is None:
                    first_exc = exc

        update_session_job(
            self.job_id,
            current_step=tasks[-1][0],
            progress_percent=WORKFLOW_STEPS[tasks[-1][0]],
            steps=self._steps_as_dicts(),
            usage=_usage_summary(self.all_usage_events),
        )

        if first_exc is not None:
            raise first_exc
        return results

    # ------------------------------------------------------------------
    # Per-step workers
    # ------------------------------------------------------------------

    def _do_analyze_jd(self) -> tuple[JDAnalysis, int, list[dict[str, Any]]]:
        ai = get_ai_service(self.effective_demo_mode)
        start = perf_counter()
        result = ai.analyze_jd(self.job_description, self.role_type, self.output_language)
        return result, int((perf_counter() - start) * 1000), _usage_events(ai)

    def _do_match_resume(self, resume_text: str) -> tuple[ResumeMatch, int, list[dict[str, Any]]]:
        ai = get_ai_service(self.effective_demo_mode)
        start = perf_counter()
        result = ai.match_resume(
            ResumeMatchRequest(
                resume_text=resume_text,
                job_description=self.job_description,
                role_type=self.role_type,
                output_language=self.output_language,
                demo_mode=self.effective_demo_mode,
            )
        )
        return result, int((perf_counter() - start) * 1000), _usage_events(ai)

    def _init_rag(
        self, jd_analysis: JDAnalysis
    ) -> tuple[RAGQuestionBank | None, list[dict[str, Any]]]:
        """Initialize the RAG question bank and retrieve few-shot examples."""
        if self.effective_demo_mode or not get_settings().openai_api_key:
            return None, []

        # Reuse the existing OpenAI client rather than creating a second HTTP connection pool.
        client = self.ai.client if isinstance(self.ai, AIInterviewService) else OpenAI(api_key=get_settings().openai_api_key)
        rag = RAGQuestionBank(client)
        
        # role_type + JD analysis 里前 10 个 required_technical_skills去搜索
        skills = [s.name for s in jd_analysis.required_technical_skills[:10]]
        examples = rag.retrieve_similar(self.role_type, skills, user_id=self.user_id)
        if examples:
            logger.info(
                "rag_retrieved",
                extra={"job_id": self.job_id, "count": len(examples), "role_type": self.role_type},
            )
        return rag, examples

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
                created_at=_now_iso(),
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

    # ------------------------------------------------------------------
    # Job lifecycle helpers
    # ------------------------------------------------------------------

    def _complete_job(self, session: SessionResponse) -> None:
        update_session_job(
            self.job_id,
            status="succeeded",
            current_step="completed",
            progress_percent=100,
            session_id=session.id if session.id else None,
            steps=self._steps_as_dicts(),
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
            steps=self._steps_as_dicts(),
            usage=_usage_summary(self.all_usage_events),
            completed=True,
        )
        logger.warning(
            "job_failed",
            extra={"job_id": self.job_id, "user_id": self.user_id, "error_code": detail["code"]},
        )

    def _mark_failed_step(self, message: str) -> None:
        if self.steps and self.steps[-1].status == "running":
            self.steps[-1].status = "failed"
            self.steps[-1].completed_at = _now_iso()
            self.steps[-1].error_message = message

    def _current_progress(self) -> int:
        if not self.steps:
            return 0
        return WORKFLOW_STEPS.get(self.steps[-1].name, 0)

    def _steps_as_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.steps]


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
