from __future__ import annotations

from openai import OpenAI

from app.evals.schemas import EvalCase, JudgeEvaluation
from app.schemas import SessionResponse


class LLMJudge:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def evaluate(self, case: EvalCase, session: SessionResponse) -> JudgeEvaluation:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator for an interview-prep LLM application. "
                        "Score only what is supported by the provided resume, job description, and generated session. "
                        "Use scores from 0.0 to 1.0. Penalize invented companies, metrics, technologies, dates, outcomes, "
                        "or claims not supported by the resume/JD."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Eval case:\n{case.model_dump_json()}\n\n"
                        f"Generated session:\n{session.model_dump_json()}\n\n"
                        "Return: faithfulness, hallucination_risk, interview_usefulness, answer_quality, and concise reasons."
                    ),
                },
            ],
            text_format=JudgeEvaluation,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("Judge did not return a parseable evaluation.")
        return parsed
