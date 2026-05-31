from __future__ import annotations

import json

from app.schemas import (
    AnswerResult,
    AnswerSet,
    JDAnalysis,
    OutputLanguage,
    QuestionSet,
    ResumeMatch,
    RoleType,
)
from app.services.llm_skills.base import LLMSkill, PromptMessages, SkillSpec


SKILL_VERSION = "2026-06-01.faithfulness-v1"
PROMPT_VERSION = SKILL_VERSION


def language_instruction(
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


def faithfulness_rules() -> str:
    return (
        "Faithfulness rules: use only facts explicitly supported by the supplied resume and job description; "
        "do not invent employers, project names, technologies, metrics, dates, certifications, degrees, or outcomes; "
        "when direct evidence is missing, clearly downgrade the claim as adjacent or preparatory experience; "
        "prefer specific quoted or near-quoted evidence over generic statements."
    )


def build_jd_analysis_prompt(
    *,
    job_description: str,
    role_type: RoleType,
    output_language: OutputLanguage,
) -> PromptMessages:
    return PromptMessages(
        system=(
            "You are a senior technical recruiter and AI interview coach. "
            "Extract concrete hiring signals from the job description only. "
            "Return only facts supported by the JD and keep items concise. "
            f"{faithfulness_rules()} "
            f"{language_instruction(output_language)}"
        ),
        user=f"Target role type: {role_type}\n\nJob description:\n{job_description}",
    )


def build_resume_match_prompt(
    *,
    resume_text: str,
    job_description: str,
    role_type: RoleType,
    output_language: OutputLanguage,
) -> PromptMessages:
    return PromptMessages(
        system=(
            "You are matching a candidate resume to a software/AI job description. "
            "Use only evidence from the resume and JD. Penalize missing must-have skills, "
            "but surface transferable strengths fairly. "
            f"{faithfulness_rules()} "
            f"{language_instruction(output_language)}"
        ),
        user=(
            f"Target role type: {role_type}\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job description:\n{job_description}"
        ),
    )


def build_question_generation_prompt(
    *,
    resume_text: str,
    job_description: str,
    role_type: RoleType,
    output_language: OutputLanguage,
    jd_context: str,
    match_context: str,
    few_shot_examples: list[dict] | None = None,
) -> PromptMessages:
    few_shot_text = ""
    if few_shot_examples:
        lines = [
            f"- {ex.get('question', '')}"
            + (f" -> {str(ex['answer'])[:200]}" if ex.get("answer") else "")
            for ex in few_shot_examples
        ]
        few_shot_text = (
            "\n\nHigh-relevance questions from similar past sessions "
            "(use as calibration reference for tone, specificity, and difficulty; do not copy verbatim):\n"
            + "\n".join(lines)
        )

    return PromptMessages(
        system=(
            "You are designing an interview preparation question bank. "
            "Generate practical questions a real interviewer may ask. "
            "Use a balanced mix of fundamentals, applied project discussion, system design, and behavior. "
            "Every question must be relevant to the supplied JD and resume, and must not imply experience absent from the resume. "
            f"{faithfulness_rules()} "
            f"{language_instruction(output_language)}"
        ),
        user=(
            f"Target role type: {role_type}\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Job description:\n{job_description}\n\n"
            f"JD analysis JSON:\n{jd_context}\n\n"
            f"Resume match JSON:\n{match_context}"
            f"{few_shot_text}"
        ),
    )


def build_answer_prompt(
    *,
    resume_text: str,
    job_description: str | None,
    role_type: RoleType,
    output_language: OutputLanguage,
    question: str,
    category: str,
    structured: bool,
) -> PromptMessages:
    language = language_instruction(
        output_language,
        fallback_language_source="question",
        job_description=job_description,
    )
    job_description_context = f"Job description:\n{job_description}\n\n" if job_description else ""
    output_rule = (
        "Return a structured answer. Each resume_evidence_used item must be a concrete fact traceable to the resume, not a generic label. "
        if structured
        else "Output only the answer text; no labels, no JSON, no markdown headers. "
    )
    return PromptMessages(
        system=(
            "You are an interview answer coach. Generate a concise, natural answer. "
            f"{output_rule}"
            f"{faithfulness_rules()} "
            f"{language}"
        ),
        user=(
            f"Target role type: {role_type}\n"
            f"Question category: {category}\n"
            f"Question: {question}\n\n"
            f"{job_description_context}"
            f"Resume:\n{resume_text}"
        ),
    )


def build_answers_prompt(
    *,
    resume_text: str,
    job_description: str,
    role_type: RoleType,
    output_language: OutputLanguage,
    questions: QuestionSet,
    category_labels: dict[str, str],
) -> PromptMessages:
    flattened: list[dict[str, str]] = []
    for category, items in (
        (category_labels["technical"], questions.technical_questions),
        (category_labels["project"], questions.project_deep_dive_questions),
        (category_labels["system_design"], questions.system_design_questions),
        (category_labels["behavioral"], questions.behavioral_questions),
    ):
        flattened.extend({"category": category, "question": item.question} for item in items)

    return PromptMessages(
        system=(
            "You are an interview answer coach. Generate concise, natural answers for every question. "
            "Return one answer for each input question, preserving category and question text exactly. "
            "Each resume_evidence_used item must be a concrete fact traceable to the resume, not a generic label. "
            f"{faithfulness_rules()} "
            f"{language_instruction(output_language, job_description=job_description)}"
        ),
        user=(
            f"Target role type: {role_type}\n\n"
            f"Job description:\n{job_description}\n\n"
            f"Questions JSON:\n{json.dumps(flattened, ensure_ascii=False)}\n\n"
            f"Resume:\n{resume_text}"
        ),
    )


JD_ANALYSIS_SKILL = LLMSkill(
    name="jd_analysis",
    version=SKILL_VERSION,
    output_schema=JDAnalysis,
    build_prompt=build_jd_analysis_prompt,
)
RESUME_MATCH_SKILL = LLMSkill(
    name="resume_match",
    version=SKILL_VERSION,
    output_schema=ResumeMatch,
    build_prompt=build_resume_match_prompt,
)
QUESTION_GENERATION_SKILL = LLMSkill(
    name="question_generation",
    version=SKILL_VERSION,
    output_schema=QuestionSet,
    build_prompt=build_question_generation_prompt,
)
ANSWER_GENERATION_SKILL = LLMSkill(
    name="answer_generation",
    version=SKILL_VERSION,
    output_schema=AnswerSet,
    build_prompt=build_answers_prompt,
)
ANSWER_STREAMING_SKILL = LLMSkill(
    name="answer_streaming",
    version=SKILL_VERSION,
    output_schema=AnswerResult,
    build_prompt=build_answer_prompt,
)


def all_skill_specs() -> list[SkillSpec]:
    return [
        JD_ANALYSIS_SKILL.spec(),
        RESUME_MATCH_SKILL.spec(),
        QUESTION_GENERATION_SKILL.spec(),
        ANSWER_GENERATION_SKILL.spec(),
        ANSWER_STREAMING_SKILL.spec(),
    ]
