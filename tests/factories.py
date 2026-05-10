from __future__ import annotations

from app.schemas import (
    AnswerResult,
    AnswerSet,
    InterviewQuestion,
    JDAnalysis,
    QuestionSet,
    ResumeMatch,
    SessionResponse,
    SkillItem,
)


RESUME_TEXT = (
    "Python backend engineer with FastAPI, SQL, REST APIs, PDF parsing, structured LLM output, "
    "and end-to-end AI workflow project experience."
)

ENGLISH_JD = (
    "We need an AI engineer with Python, backend API design, SQL persistence, LLM application "
    "experience, system design judgment, and the ability to explain project tradeoffs."
)

CHINESE_JD = (
    "我们需要一名 AI 工程师，熟悉 Python、后端 API、LLM 应用、系统设计、数据库持久化、"
    "结构化输出、提示词工程，并且能够独立交付端到端的 AI 产品功能。"
)


def sample_jd_analysis() -> JDAnalysis:
    return JDAnalysis(
        role_summary="Backend AI role focused on production LLM systems.",
        required_technical_skills=[
            SkillItem(name="Python", importance="must_have", evidence="Python listed as required."),
            SkillItem(name="API design", importance="must_have", evidence="APIs listed in JD."),
        ],
        preferred_skills=[SkillItem(name="SQL", importance="preferred", evidence="Persistence mentioned.")],
        likely_interview_topics=["API design", "LLM workflows"],
        backend_system_design_topics=["Persistence", "Service boundaries"],
        ai_llm_topics=["Structured outputs", "Hallucination control"],
        seniority_signals=["Can own end-to-end delivery"],
        red_flags_or_ambiguities=[],
    )


def sample_resume_match() -> ResumeMatch:
    return ResumeMatch(
        overall_fit_score=82,
        strong_matches=[
            {
                "skill": "Python",
                "resume_evidence": "Resume includes Python backend project experience.",
                "strength": "strong",
            }
        ],
        missing_skills=["Kubernetes"],
        transferable_strengths=["Backend APIs", "AI workflow delivery"],
        suggested_positioning_strategy="Lead with end-to-end AI workflow delivery.",
        recommended_project_talking_points=["Interview Copilot architecture", "Structured output reliability"],
        resume_risks_to_prepare_for=["Deployment depth"],
    )


def sample_questions() -> QuestionSet:
    return QuestionSet(
        technical_questions=[
            InterviewQuestion(
                question="How do you make LLM output reliable enough for application code?",
                why_it_matters="Tests structured output and validation.",
                difficulty="hard",
            )
        ],
        project_deep_dive_questions=[
            InterviewQuestion(
                question="What was the most important engineering tradeoff in this project?",
                why_it_matters="Tests project ownership.",
                difficulty="medium",
            )
        ],
        system_design_questions=[
            InterviewQuestion(
                question="How would you scale saved interview sessions?",
                why_it_matters="Tests persistence and system design.",
                difficulty="medium",
            )
        ],
        behavioral_questions=[
            InterviewQuestion(
                question="Tell me about a time you learned a new technology quickly.",
                why_it_matters="Tests learning speed and communication.",
                difficulty="warmup",
            )
        ],
    )


def sample_answers() -> AnswerSet:
    return AnswerSet(
        answers=[
            AnswerResult(
                category="Technical",
                question="How do you make LLM output reliable enough for application code?",
                concise_answer="I use Pydantic schemas, structured model output, and validation before display.",
                resume_evidence_used=["Interview Copilot project"],
                honesty_guardrail="Do not invent production scale that is not in the resume.",
            )
        ]
    )


def sample_session(session_id: int = 1) -> SessionResponse:
    return SessionResponse(
        id=session_id,
        created_at="2026-05-11T00:00:00+00:00",
        role_type="AI Engineer",
        output_language="English",
        demo_mode=False,
        job_description=ENGLISH_JD,
        resume_text=RESUME_TEXT,
        jd_analysis=sample_jd_analysis(),
        resume_match=sample_resume_match(),
        questions=sample_questions(),
        answers=sample_answers(),
    )
