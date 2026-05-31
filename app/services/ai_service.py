from __future__ import annotations

import json
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


class AIInterviewService:
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
        return self._parse(
            JDAnalysis,
            system=(
                "You are a senior technical recruiter and AI interview coach. "
                "Extract concrete hiring signals from the job description. "
                "Return only facts supported by the JD and keep items concise. "
                f"{self._language_instruction(output_language)}"
            ),
            user=f"Target role type: {role_type}\n\nJob description:\n{job_description}",
        )

    def match_resume(self, request: ResumeMatchRequest) -> ResumeMatch:
        return self._parse(
            ResumeMatch,
            system=(
                "You are matching a candidate resume to a software/AI job description. "
                "Use only evidence from the resume and JD. Penalize missing must-have skills, "
                "but surface transferable strengths fairly. "
                f"{self._language_instruction(request.output_language)}"
            ),
            user=(
                f"Target role type: {request.role_type}\n\n"
                f"Resume:\n{request.resume_text}\n\n"
                f"Job description:\n{request.job_description}"
            ),
        )

    def generate_questions(self, request: GenerateQuestionsRequest) -> QuestionSet:
        jd_context = request.jd_analysis.model_dump_json() if request.jd_analysis else "Not provided."
        match_context = request.resume_match.model_dump_json() if request.resume_match else "Not provided."
        return self._parse(
            QuestionSet,
            system=(
                "You are designing an interview preparation question bank. "
                "Generate practical questions a real interviewer may ask. "
                "Use a balanced mix of fundamentals, applied project discussion, system design, and behavior. "
                f"{self._language_instruction(request.output_language)}"
            ),
            user=(
                f"Target role type: {request.role_type}\n\n"
                f"Resume:\n{request.resume_text}\n\n"
                f"Job description:\n{request.job_description}\n\n"
                f"JD analysis JSON:\n{jd_context}\n\n"
                f"Resume match JSON:\n{match_context}"
            ),
        )

    def generate_answer(self, request: GenerateAnswerRequest) -> AnswerResult:
        language_instruction = self._language_instruction(
            request.output_language,
            fallback_language_source="question",
            job_description=request.job_description,
        )
        job_description_context = (
            f"Job description:\n{request.job_description}\n\n" if request.job_description else ""
        )
        return self._parse(
            AnswerResult,
            system=(
                "You are an interview answer coach. Generate a concise, natural answer. "
                "Use only the resume information supplied by the user. "
                "Do not invent employers, project names, technologies, metrics, dates, or outcomes. "
                "If the resume lacks direct evidence, say how to honestly frame adjacent experience. "
                f"{language_instruction}"
            ),
            user=(
                f"Target role type: {request.role_type}\n"
                f"Question category: {request.category}\n"
                f"Question: {request.question}\n\n"
                f"{job_description_context}"
                f"Resume:\n{request.resume_text}"
            ),
        )

    def stream_answer(self, request: GenerateAnswerRequest) -> Iterator[str]:
        language_instruction = self._language_instruction(
            request.output_language,
            fallback_language_source="question",
            job_description=request.job_description,
        )
        job_description_context = (
            f"Job description:\n{request.job_description}\n\n" if request.job_description else ""
        )
        with self.client.responses.stream(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an interview answer coach. Write a concise, natural answer. "
                        "Output only the answer text — no labels, no JSON, no markdown headers. "
                        "Use only the resume information supplied. "
                        "Do not invent employers, project names, technologies, metrics, dates, or outcomes. "
                        "If the resume lacks direct evidence, say how to honestly frame adjacent experience. "
                        f"{language_instruction}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target role type: {request.role_type}\n"
                        f"Question category: {request.category}\n"
                        f"Question: {request.question}\n\n"
                        f"{job_description_context}"
                        f"Resume:\n{request.resume_text}"
                    ),
                },
            ],
        ) as stream:
            yield from stream.text_deltas

    def generate_answers_for_question_set(
        self,
        *,
        resume_text: str,
        job_description: str,
        role_type: RoleType,
        output_language: OutputLanguage,
        questions: QuestionSet,
    ) -> AnswerSet:
        category_labels = self._answer_category_labels(output_language, job_description)
        flattened: list[dict[str, str]] = []
        for category, items in (
            (category_labels["technical"], questions.technical_questions),
            (category_labels["project"], questions.project_deep_dive_questions),
            (category_labels["system_design"], questions.system_design_questions),
            (category_labels["behavioral"], questions.behavioral_questions),
        ):
            flattened.extend({"category": category, "question": item.question} for item in items)

        return self._parse(
            AnswerSet,
            system=(
                "You are an interview answer coach. Generate concise, natural answers for every question. "
                "Use only the resume information supplied by the user. "
                "Do not invent employers, project names, technologies, metrics, dates, or outcomes. "
                "If the resume lacks direct evidence, say how to honestly frame adjacent experience. "
                "Return one answer for each input question, preserving category and question text exactly. "
                f"{self._language_instruction(output_language, job_description=job_description)}"
            ),
            user=(
                f"Target role type: {role_type}\n\n"
                f"Job description:\n{job_description}\n\n"
                f"Questions JSON:\n{json.dumps(flattened, ensure_ascii=False)}\n\n"
                f"Resume:\n{resume_text}"
            ),
        )

    def _parse(self, schema: type, *, system: str, user: str):
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
        if output_language == "English":
            return "Write every user-facing field in English."
        if output_language == "Chinese":
            return "Write every user-facing field in Chinese."
        if job_description:
            return (
                "Write every user-facing field in the same language as the job description. "
                "If the job description is Chinese, write in Chinese and avoid English category labels unless they are proper nouns or technical terms."
            )
        return f"Write every user-facing field in the same language as the {fallback_language_source}."

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
