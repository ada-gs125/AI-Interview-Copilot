from __future__ import annotations

import re
from statistics import mean

from app.evals.schemas import EvalCase, RuleEvaluation, ScoreBreakdown
from app.schemas import AnswerResult, InterviewQuestion, SessionResponse


_WORD_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _flatten_questions(session: SessionResponse) -> list[InterviewQuestion]:
    questions = session.questions
    return (
        questions.technical_questions
        + questions.project_deep_dive_questions
        + questions.system_design_questions
        + questions.behavioral_questions
    )


def _all_answer_text(answers: list[AnswerResult]) -> str:
    return "\n".join(
        [
            answer.concise_answer
            + "\n"
            + "\n".join(answer.resume_evidence_used)
            + "\n"
            + answer.honesty_guardrail
            for answer in answers
        ]
    )


def _score_from_failures(failures: list[str]) -> float:
    return 1.0 if not failures else max(0.0, 1.0 - 0.25 * len(failures))


def _score_schema(session: SessionResponse) -> ScoreBreakdown:
    return ScoreBreakdown(score=1.0, reasons=["Session parsed as SessionResponse."])


def _score_question_completeness(session: SessionResponse) -> ScoreBreakdown:
    questions = session.questions
    counts = {
        "technical": len(questions.technical_questions),
        "project": len(questions.project_deep_dive_questions),
        "system_design": len(questions.system_design_questions),
        "behavioral": len(questions.behavioral_questions),
    }
    failures = [f"{name} has no questions" for name, count in counts.items() if count == 0]
    total = sum(counts.values())
    if total < 4:
        failures.append(f"only {total} total questions")
    if total > 24:
        failures.append(f"{total} total questions is likely too many")
    return ScoreBreakdown(score=_score_from_failures(failures), reasons=failures or [f"{total} questions across all categories."])


def _score_duplicates(session: SessionResponse) -> ScoreBreakdown:
    normalized = [_norm(q.question) for q in _flatten_questions(session)]
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in normalized:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    total = max(1, len(normalized))
    duplicate_ratio = len(duplicates) / total
    reasons = [f"{len(duplicates)} duplicate questions."] if duplicates else ["No duplicate questions."]
    return ScoreBreakdown(score=max(0.0, 1.0 - duplicate_ratio), reasons=reasons)


def _score_language(case: EvalCase, session: SessionResponse) -> ScoreBreakdown:
    expected = case.output_language
    if expected == "Match job description language":
        expected = "Chinese" if _has_chinese(case.job_description) else "English"
    text = (
        session.jd_analysis.role_summary
        + "\n"
        + "\n".join(q.question for q in _flatten_questions(session))
        + "\n"
        + _all_answer_text(session.answers.answers)
    )
    if expected == "Chinese":
        score = 1.0 if _has_chinese(text) else 0.0
        reason = "Chinese output detected." if score else "Expected Chinese output but no Chinese text was detected."
        return ScoreBreakdown(score=score, reasons=[reason])
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ratio = chinese_chars / max(1, len(text))
    score = 1.0 if ratio < 0.02 else 0.4
    reason = "English output detected." if score == 1.0 else "Expected English output but Chinese text appeared."
    return ScoreBreakdown(score=score, reasons=[reason])


def _score_answer_alignment(session: SessionResponse) -> ScoreBreakdown:
    questions = {q.question for q in _flatten_questions(session)}
    answers = session.answers.answers
    matched = sum(1 for answer in answers if answer.question in questions)
    failures: list[str] = []
    if len(answers) != len(questions):
        failures.append(f"{len(answers)} answers for {len(questions)} questions")
    if matched != len(answers):
        failures.append(f"{len(answers) - matched} answers do not match generated questions")
    score = matched / max(1, max(len(answers), len(questions)))
    if failures:
        score = min(score, _score_from_failures(failures))
    return ScoreBreakdown(score=score, reasons=failures or ["Every answer maps to a generated question."])


def _score_empty_fields(session: SessionResponse) -> ScoreBreakdown:
    failures: list[str] = []
    for answer in session.answers.answers:
        if len(answer.concise_answer.strip()) < 40:
            failures.append(f"answer too short: {answer.question[:60]}")
        if not answer.resume_evidence_used:
            failures.append(f"missing resume evidence: {answer.question[:60]}")
        if not answer.honesty_guardrail.strip():
            failures.append(f"missing honesty guardrail: {answer.question[:60]}")
    for question in _flatten_questions(session):
        if len(question.question.strip()) < 10:
            failures.append("question text too short")
        if not question.why_it_matters.strip():
            failures.append(f"missing why_it_matters: {question.question[:60]}")
    return ScoreBreakdown(score=_score_from_failures(failures), reasons=failures or ["No empty or obviously short fields."])


def _score_forbidden_claims(case: EvalCase, session: SessionResponse) -> ScoreBreakdown:
    output = _norm(
        session.jd_analysis.model_dump_json()
        + "\n"
        + session.resume_match.model_dump_json()
        + "\n"
        + session.questions.model_dump_json()
        + "\n"
        + session.answers.model_dump_json()
    )
    failures = [claim for claim in case.forbidden_claims if _norm(claim) and _norm(claim) in output]
    return ScoreBreakdown(
        score=1.0 if not failures else 0.0,
        reasons=[f"Forbidden claim found: {claim}" for claim in failures] or ["No forbidden claims found."],
    )


def _score_evidence_grounding(case: EvalCase, session: SessionResponse) -> ScoreBreakdown:
    resume = _norm(case.resume_text)
    generic_markers = ("demo mode", "sample evidence", "resume evidence", "relevant experience")
    failures: list[str] = []
    for answer in session.answers.answers:
        for evidence in answer.resume_evidence_used:
            normalized = _norm(evidence)
            if not normalized or any(marker in normalized for marker in generic_markers):
                failures.append(f"generic evidence: {evidence[:80]}")
                continue
            tokens = [token for token in normalized.split() if len(token) >= 4]
            if tokens and not any(token in resume for token in tokens):
                failures.append(f"evidence not traceable to resume: {evidence[:80]}")
    return ScoreBreakdown(score=_score_from_failures(failures), reasons=failures or ["Resume evidence appears traceable."])


def evaluate_rules(case: EvalCase, session: SessionResponse) -> RuleEvaluation:
    categories = {
        "schema": _score_schema(session),
        "question_completeness": _score_question_completeness(session),
        "duplicates": _score_duplicates(session),
        "language": _score_language(case, session),
        "answer_alignment": _score_answer_alignment(session),
        "empty_fields": _score_empty_fields(session),
        "forbidden_claims": _score_forbidden_claims(case, session),
        "evidence_grounding": _score_evidence_grounding(case, session),
    }
    score = mean(item.score for item in categories.values())
    failures = [
        reason
        for item in categories.values()
        if item.score < 1.0
        for reason in item.reasons
    ]
    return RuleEvaluation(score=round(score, 4), categories=categories, failures=failures)
