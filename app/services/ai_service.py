import json

from openai import OpenAI

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
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("The model did not return a parseable structured response.")
        return parsed

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
