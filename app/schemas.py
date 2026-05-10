from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


RoleType = Literal[
    "Backend Engineer",
    "AI Engineer",
    "Agent Engineer",
    "Full-stack AI Engineer",
]

OutputLanguage = Literal["English", "Chinese", "Match job description language"]


class AnalyzeJDRequest(BaseModel):
    job_description: str = Field(min_length=50)
    role_type: RoleType
    output_language: OutputLanguage = "Match job description language"
    demo_mode: bool = False


class SkillItem(BaseModel):
    name: str
    importance: Literal["must_have", "preferred", "nice_to_have"]
    evidence: str


class JDAnalysis(BaseModel):
    role_summary: str
    required_technical_skills: list[SkillItem]
    preferred_skills: list[SkillItem]
    likely_interview_topics: list[str]
    backend_system_design_topics: list[str]
    ai_llm_topics: list[str]
    seniority_signals: list[str]
    red_flags_or_ambiguities: list[str]


class ResumeMatchRequest(BaseModel):
    resume_text: str = Field(min_length=50)
    job_description: str = Field(min_length=50)
    role_type: RoleType
    output_language: OutputLanguage = "Match job description language"
    demo_mode: bool = False


class EvidenceMatch(BaseModel):
    skill: str
    resume_evidence: str
    strength: Literal["strong", "partial", "weak"]


class ResumeMatch(BaseModel):
    overall_fit_score: int = Field(ge=0, le=100)
    strong_matches: list[EvidenceMatch]
    missing_skills: list[str]
    transferable_strengths: list[str]
    suggested_positioning_strategy: str
    recommended_project_talking_points: list[str]
    resume_risks_to_prepare_for: list[str]


class GenerateQuestionsRequest(BaseModel):
    resume_text: str = Field(min_length=50)
    job_description: str = Field(min_length=50)
    role_type: RoleType
    output_language: OutputLanguage = "Match job description language"
    demo_mode: bool = False
    jd_analysis: Optional[JDAnalysis] = None
    resume_match: Optional[ResumeMatch] = None


class InterviewQuestion(BaseModel):
    question: str
    why_it_matters: str
    difficulty: Literal["warmup", "medium", "hard"]


class QuestionSet(BaseModel):
    technical_questions: list[InterviewQuestion]
    project_deep_dive_questions: list[InterviewQuestion]
    system_design_questions: list[InterviewQuestion]
    behavioral_questions: list[InterviewQuestion]


class GenerateAnswerRequest(BaseModel):
    resume_text: str = Field(min_length=50)
    role_type: RoleType
    output_language: OutputLanguage = "Match job description language"
    demo_mode: bool = False
    question: str = Field(min_length=10)
    category: str = Field(min_length=3)


class AnswerResult(BaseModel):
    category: str
    question: str
    concise_answer: str
    resume_evidence_used: list[str]
    honesty_guardrail: str


class AnswerSet(BaseModel):
    answers: list[AnswerResult]


class SessionResponse(BaseModel):
    id: int
    created_at: str
    role_type: RoleType
    output_language: OutputLanguage
    demo_mode: bool = False
    job_description: str
    resume_text: str
    jd_analysis: JDAnalysis
    resume_match: ResumeMatch
    questions: QuestionSet
    answers: AnswerSet


class SessionSummary(BaseModel):
    id: int
    created_at: str
    role_type: RoleType
    output_language: OutputLanguage
    demo_mode: bool = False
    overall_fit_score: int
    role_summary: str
    missing_skill_count: int
