from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas import OutputLanguage, RoleType, SessionResponse


class EvalCase(BaseModel):
    id: str
    resume_text: str = Field(min_length=50)
    job_description: str = Field(min_length=50)
    role_type: RoleType
    output_language: OutputLanguage = "English"
    expected_facts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class RuleEvaluation(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    categories: dict[str, ScoreBreakdown]
    failures: list[str] = Field(default_factory=list)


class JudgeEvaluation(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    hallucination_risk: float = Field(ge=0.0, le=1.0)
    interview_usefulness: float = Field(ge=0.0, le=1.0)
    answer_quality: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class CaseEvaluation(BaseModel):
    case_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    weighted_scores: dict[str, float]
    rule_evaluation: RuleEvaluation
    judge_evaluation: JudgeEvaluation | None = None
    failures: list[str] = Field(default_factory=list)
    session: SessionResponse | None = None


class EvalSummary(BaseModel):
    case_count: int
    average_score: float = Field(ge=0.0, le=1.0)
    prompt_version: str
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    summary: EvalSummary
    cases: list[CaseEvaluation]
