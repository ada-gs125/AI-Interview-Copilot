from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable

from openai import OpenAI

from app.config import get_settings
from app.database import attach_session_job_evaluation, init_db
from app.evals.judge import LLMJudge
from app.evals.rules import evaluate_rules
from app.evals.schemas import CaseEvaluation, EvalCase, EvalReport, EvalSummary, JudgeEvaluation
from app.schemas import (
    AnswerSet,
    GenerateQuestionsRequest,
    ResumeMatchRequest,
    SessionResponse,
)
from app.services.ai_service import AIInterviewService
from app.services.interview_workflow import _usage_summary
from app.services.llm_skills import all_skill_specs
from app.services.mock_service import MockAIInterviewService
from app.services.prompts import PROMPT_VERSION


def load_dataset(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cases.append(EvalCase.model_validate_json(stripped))
            except Exception as exc:
                raise ValueError(f"Invalid eval case at {path}:{line_no}: {exc}") from exc
    if not cases:
        raise ValueError(f"No eval cases found in {path}.")
    return cases


def _generate_session(case: EvalCase, ai) -> SessionResponse:
    jd_analysis = ai.analyze_jd(case.job_description, case.role_type, case.output_language)
    resume_match = ai.match_resume(
        ResumeMatchRequest(
            resume_text=case.resume_text,
            job_description=case.job_description,
            role_type=case.role_type,
            output_language=case.output_language,
        )
    )
    questions = ai.generate_questions(
        GenerateQuestionsRequest(
            resume_text=case.resume_text,
            job_description=case.job_description,
            role_type=case.role_type,
            output_language=case.output_language,
            jd_analysis=jd_analysis,
            resume_match=resume_match,
        )
    )
    answers = ai.generate_answers_for_question_set(
        resume_text=case.resume_text,
        job_description=case.job_description,
        role_type=case.role_type,
        output_language=case.output_language,
        questions=questions,
    )
    return SessionResponse(
        id=0,
        created_at=datetime.now(timezone.utc).isoformat(),
        role_type=case.role_type,
        output_language=case.output_language,
        demo_mode=isinstance(ai, MockAIInterviewService),
        job_description=case.job_description,
        resume_text=case.resume_text,
        jd_analysis=jd_analysis,
        resume_match=resume_match,
        questions=questions,
        answers=answers if isinstance(answers, AnswerSet) else AnswerSet.model_validate(answers),
    )


def _combine_scores(rule_score: float, judge: JudgeEvaluation | None) -> tuple[float, dict[str, float]]:
    if judge is None:
        weighted = {
            "truthfulness": rule_score,
            "interview_usefulness": rule_score,
            "language_and_structure": rule_score,
            "completeness": rule_score,
        }
    else:
        truthfulness = mean([judge.faithfulness, 1.0 - judge.hallucination_risk])
        weighted = {
            "truthfulness": truthfulness,
            "interview_usefulness": judge.interview_usefulness,
            "language_and_structure": rule_score,
            "completeness": mean([rule_score, judge.answer_quality]),
        }
    overall = (
        weighted["truthfulness"] * 0.50
        + weighted["interview_usefulness"] * 0.25
        + weighted["language_and_structure"] * 0.15
        + weighted["completeness"] * 0.10
    )
    return round(overall, 4), {key: round(value, 4) for key, value in weighted.items()}


def evaluate_case(case: EvalCase, ai, judge: LLMJudge | None = None) -> CaseEvaluation:
    session = _generate_session(case, ai)
    rule_evaluation = evaluate_rules(case, session)
    judge_evaluation = judge.evaluate(case, session) if judge is not None else None
    overall_score, weighted_scores = _combine_scores(rule_evaluation.score, judge_evaluation)
    failures = list(rule_evaluation.failures)
    if judge_evaluation and judge_evaluation.hallucination_risk >= 0.25:
        failures.append(f"Judge hallucination risk is {judge_evaluation.hallucination_risk:.2f}.")
    return CaseEvaluation(
        case_id=case.id,
        overall_score=overall_score,
        weighted_scores=weighted_scores,
        rule_evaluation=rule_evaluation,
        judge_evaluation=judge_evaluation,
        failures=failures,
        session=session,
    )


def run_evaluation(
    cases: Iterable[EvalCase],
    *,
    use_mock_ai: bool = False,
    use_llm_judge: bool = True,
) -> EvalReport:
    settings = get_settings()
    ai = MockAIInterviewService() if use_mock_ai else AIInterviewService(settings)
    judge = None
    if use_llm_judge and settings.openai_api_key:
        judge = LLMJudge(OpenAI(api_key=settings.openai_api_key), settings.openai_model)

    case_results = [evaluate_case(case, ai, judge=judge) for case in cases]
    usage = _usage_summary(ai.usage_snapshot()) if hasattr(ai, "usage_snapshot") else {"call_count": 0, "calls": []}
    usage["llm_judge_enabled"] = judge is not None
    usage["mock_ai"] = use_mock_ai
    average_score = mean(result.overall_score for result in case_results) if case_results else 0.0
    return EvalReport(
        summary=EvalSummary(
            case_count=len(case_results),
            average_score=round(average_score, 4),
            prompt_version=PROMPT_VERSION,
            skill_versions=[dataclasses.asdict(spec) for spec in all_skill_specs()],
            model=settings.openai_model if not use_mock_ai else "mock-ai",
            usage=usage,
        ),
        cases=case_results,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local LLM output evaluations.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to a JSONL eval dataset.")
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report.")
    parser.add_argument("--mock-ai", action="store_true", help="Use deterministic mock AI outputs.")
    parser.add_argument("--skip-llm-judge", action="store_true", help="Run only rule-based evaluation.")
    parser.add_argument("--job-id", help="Attach report summary to session_jobs.usage.evaluation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_evaluation(
        load_dataset(args.dataset),
        use_mock_ai=args.mock_ai,
        use_llm_judge=not args.skip_llm_judge,
    )
    payload = report.model_dump(mode="json")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.job_id:
        init_db()
        attached = attach_session_job_evaluation(args.job_id, payload["summary"])
        if not attached:
            print(f"Job not found: {args.job_id}", file=sys.stderr)
            return 2

    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
