"""OpenAI-backed interview analysis and answer generation service."""

from __future__ import annotations

from typing import Any, Iterator

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from app.config import Settings
from app.schemas import (
    AnswerResult,
    AnswerSet,
    GenerateAnswerRequest,
    GenerateQuestionsRequest,
    JDAnalysis,
    QuestionSet,
    ResumeMatch,
    ResumeMatchRequest,
    OutputLanguage,
    RoleType,
)
from app.services.prompts import (
    generate_answer_prompt,
    generate_answers_prompt,
    generate_questions_prompt,
    jd_analysis_prompt,
    language_instruction,
    resume_match_prompt,
)


class AIInterviewService:
    """Calls OpenAI Responses API and returns validated Pydantic models."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. Add it to .env before running AI workflows.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.input_cost_per_1m_tokens = settings.openai_input_cost_per_1m_tokens
        self.output_cost_per_1m_tokens = settings.openai_output_cost_per_1m_tokens
        self.usage_events: list[dict[str, Any]] = []

    def analyze_jd(
        self,
        job_description: str,
        role_type: RoleType,
        output_language: OutputLanguage = "Match job description language",
    ) -> JDAnalysis:
        # Extract hiring signals from the job description only.
        prompt = jd_analysis_prompt(
            job_description=job_description,
            role_type=role_type,
            output_language=output_language,
        )
        return self._parse(
            JDAnalysis,
            system=prompt.system,
            user=prompt.user,
        )

    def match_resume(self, request: ResumeMatchRequest) -> ResumeMatch:
        # Compare resume evidence against the job description without inventing facts.
        prompt = resume_match_prompt(
            resume_text=request.resume_text,
            job_description=request.job_description,
            role_type=request.role_type,
            output_language=request.output_language,
        )
        return self._parse(
            ResumeMatch,
            system=prompt.system,
            user=prompt.user,
        )

    def generate_questions(
        self,
        request: GenerateQuestionsRequest,
        *,
        few_shot_examples: list[dict] | None = None,
    ) -> QuestionSet:
        jd_context = request.jd_analysis.model_dump_json() if request.jd_analysis else "Not provided."
        match_context = request.resume_match.model_dump_json() if request.resume_match else "Not provided."
        prompt = generate_questions_prompt(
            resume_text=request.resume_text,
            job_description=request.job_description,
            role_type=request.role_type,
            output_language=request.output_language,
            jd_context=jd_context,
            match_context=match_context,
            few_shot_examples=few_shot_examples,
        )

        return self._parse(
            QuestionSet,
            system=prompt.system,
            user=prompt.user,
        )

    def generate_answer(self, request: GenerateAnswerRequest) -> AnswerResult:
        # Generate one grounded answer for a selected interview question.
        prompt = generate_answer_prompt(
            resume_text=request.resume_text,
            job_description=request.job_description,
            role_type=request.role_type,
            output_language=request.output_language,
            question=request.question,
            category=request.category,
            structured=True,
        )
        return self._parse(
            AnswerResult,
            system=prompt.system,
            user=prompt.user,
        )

    def stream_answer(self, request: GenerateAnswerRequest) -> Iterator[str]:
        # Used by the frontend regenerate button for live token streaming.
        prompt = generate_answer_prompt(
            resume_text=request.resume_text,
            job_description=request.job_description,
            role_type=request.role_type,
            output_language=request.output_language,
            question=request.question,
            category=request.category,
            structured=False,
        )
        with self.client.responses.stream(
            model=self.model,
            input=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        ) as stream:
            if hasattr(stream, "text_deltas"):
                yield from stream.text_deltas
                return

            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    yield event.delta

    def generate_answers_for_question_set(
        self,
        *,
        resume_text: str,
        job_description: str,
        role_type: RoleType,
        output_language: OutputLanguage,
        questions: QuestionSet,
    ) -> AnswerSet:
        # Batch answers preserve each generated question and category.
        category_labels = self._answer_category_labels(output_language, job_description)
        prompt = generate_answers_prompt(
            resume_text=resume_text,
            job_description=job_description,
            role_type=role_type,
            output_language=output_language,
            questions=questions,
            category_labels=category_labels,
        )

        return self._parse(
            AnswerSet,
            system=prompt.system,
            user=prompt.user,
        )

    def _parse(self, schema: type, *, system: str, user: str):
        # Structured outputs keep downstream code from parsing raw JSON manually.
        response = self._call_api(schema, system=system, user=user)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("The model did not return a parseable structured response.")
        self._record_usage(response, schema)
        return parsed

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=2, max=30),
        reraise=True,
    )
    def _call_api(self, schema: type, *, system: str, user: str):
        # Retry transient model/API failures with bounded exponential backoff.
        return self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=schema,
        )

    def usage_snapshot(self) -> list[dict[str, Any]]:
        return list(getattr(self, "usage_events", []))

    def _record_usage(self, response: Any, schema: type) -> None:
        # Store per-call token usage so the job progress UI can show totals.
        usage = getattr(response, "usage", None)
        event: dict[str, Any] = {
            "schema": schema.__name__,
            "model": self.model,
        }
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            event.update(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            )
            estimated_cost = self._estimated_cost_usd(input_tokens, output_tokens)
            if estimated_cost is not None:
                event["estimated_cost_usd"] = estimated_cost
        self.usage_events.append(event)

    def _estimated_cost_usd(self, input_tokens: Any, output_tokens: Any) -> float | None:
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            return None
        if not self.input_cost_per_1m_tokens and not self.output_cost_per_1m_tokens:
            return None
        return round(
            (input_tokens / 1_000_000 * self.input_cost_per_1m_tokens)
            + (output_tokens / 1_000_000 * self.output_cost_per_1m_tokens),
            6,
        )

    def _language_instruction(
        self,
        output_language: OutputLanguage,
        *,
        fallback_language_source: str = "job description",
        job_description: str | None = None,
    ) -> str:
        return language_instruction(
            output_language,
            fallback_language_source=fallback_language_source,
            job_description=job_description,
        )

    def _answer_category_labels(
        self,
        output_language: OutputLanguage,
        job_description: str,
    ) -> dict[str, str]:
        if output_language == "Chinese" or (
            output_language == "Match job description language" and self._looks_chinese(job_description)
        ):
            return {
                "technical": "技术问题",
                "project": "项目深挖问题",
                "system_design": "系统设计问题",
                "behavioral": "行为面试问题",
            }
        return {
            "technical": "Technical",
            "project": "Project Deep-Dive",
            "system_design": "System Design",
            "behavioral": "Behavioral",
        }

    def _looks_chinese(self, text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)
