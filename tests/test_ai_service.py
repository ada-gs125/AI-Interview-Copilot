from types import SimpleNamespace

from app.services.ai_service import AIInterviewService
from app.schemas import GenerateAnswerRequest

from tests.factories import CHINESE_JD, ENGLISH_JD, RESUME_TEXT


def test_answer_category_labels_match_chinese_jd_language():
    service = object.__new__(AIInterviewService)

    labels = service._answer_category_labels("Match job description language", CHINESE_JD)

    assert labels == {
        "technical": "技术问题",
        "project": "项目深挖问题",
        "system_design": "系统设计问题",
        "behavioral": "行为面试问题",
    }


def test_answer_category_labels_stay_english_for_english_jd():
    service = object.__new__(AIInterviewService)

    labels = service._answer_category_labels("Match job description language", ENGLISH_JD)

    assert labels["technical"] == "Technical"
    assert labels["system_design"] == "System Design"


def test_language_instruction_is_explicit_when_matching_chinese_jd():
    service = object.__new__(AIInterviewService)

    instruction = service._language_instruction(
        "Match job description language",
        job_description=CHINESE_JD,
    )

    assert "same language as the job description" in instruction
    assert "If the job description is Chinese" in instruction


def test_estimated_cost_uses_configured_token_prices():
    service = object.__new__(AIInterviewService)
    service.input_cost_per_1m_tokens = 0.25
    service.output_cost_per_1m_tokens = 2.0

    assert service._estimated_cost_usd(1_000_000, 500_000) == 1.25


def test_estimated_cost_is_omitted_without_prices():
    service = object.__new__(AIInterviewService)
    service.input_cost_per_1m_tokens = 0.0
    service.output_cost_per_1m_tokens = 0.0

    assert service._estimated_cost_usd(100, 200) is None


def test_stream_answer_reads_openai_response_delta_events():
    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def __iter__(self):
            return iter(
                [
                    SimpleNamespace(type="response.created", delta="ignored"),
                    SimpleNamespace(type="response.output_text.delta", delta="hello"),
                    SimpleNamespace(type="response.output_text.delta", delta=" world"),
                ]
            )

    class FakeResponses:
        def stream(self, **kwargs):
            assert kwargs["model"] == "gpt-test"
            return FakeStream()

    service = object.__new__(AIInterviewService)
    service.client = SimpleNamespace(responses=FakeResponses())
    service.model = "gpt-test"

    tokens = list(
        service.stream_answer(
            GenerateAnswerRequest(
                resume_text=RESUME_TEXT,
                job_description=ENGLISH_JD,
                role_type="AI Engineer",
                output_language="English",
                question="How do you make LLM output reliable for application code?",
                category="Technical",
            )
        )
    )

    assert tokens == ["hello", " world"]
