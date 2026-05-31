from pathlib import Path

from app.evals.rules import evaluate_rules
from app.evals.run import load_dataset, run_evaluation
from app.evals.schemas import EvalCase
from app.schemas import AnswerResult, AnswerSet, InterviewQuestion, QuestionSet
from tests.factories import ENGLISH_JD, RESUME_TEXT, sample_jd_analysis, sample_resume_match, sample_session


def _case(**overrides) -> EvalCase:
    payload = {
        "id": "case-1",
        "resume_text": RESUME_TEXT,
        "job_description": ENGLISH_JD,
        "role_type": "AI Engineer",
        "output_language": "English",
        "expected_facts": ["FastAPI"],
        "forbidden_claims": ["Kubernetes"],
    }
    payload.update(overrides)
    return EvalCase.model_validate(payload)


def test_rule_evaluator_detects_hallucination_and_answer_mismatch():
    session = sample_session()
    session.answers = AnswerSet(
        answers=[
            AnswerResult(
                category="Technical",
                question="A question that was not generated",
                concise_answer="I ran a Kubernetes production cluster for millions of users.",
                resume_evidence_used=["Kubernetes production cluster"],
                honesty_guardrail="No caveat.",
            )
        ]
    )

    result = evaluate_rules(_case(), session)

    assert result.score < 1.0
    assert result.categories["forbidden_claims"].score == 0.0
    assert result.categories["answer_alignment"].score < 1.0
    assert any("Forbidden claim" in failure for failure in result.failures)


def test_rule_evaluator_detects_duplicate_questions_and_language_mismatch():
    session = sample_session()
    question = InterviewQuestion(question="同一个问题是什么？", why_it_matters="测试重复。", difficulty="medium")
    session.questions = QuestionSet(
        technical_questions=[question],
        project_deep_dive_questions=[question],
        system_design_questions=[],
        behavioral_questions=[],
    )

    result = evaluate_rules(_case(output_language="English"), session)

    assert result.categories["duplicates"].score < 1.0
    assert result.categories["question_completeness"].score < 1.0
    assert result.categories["language"].score < 1.0


def test_eval_runner_loads_dataset_and_runs_with_mock_ai():
    cases = load_dataset(Path("tests/evals/golden_sessions.jsonl"))

    report = run_evaluation(cases, use_mock_ai=True, use_llm_judge=False)

    assert report.summary.case_count == 1
    assert report.summary.model == "mock-ai"
    assert report.summary.prompt_version
    assert report.cases[0].session is not None


def test_rule_evaluator_accepts_good_fixture_session():
    session = sample_session()
    session.jd_analysis = sample_jd_analysis()
    session.resume_match = sample_resume_match()

    result = evaluate_rules(_case(forbidden_claims=["AWS SageMaker"]), session)

    assert result.categories["schema"].score == 1.0
    assert result.categories["forbidden_claims"].score == 1.0
