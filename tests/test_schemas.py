from app.schemas import GenerateQuestionsRequest, JDAnalysis, SkillItem
from app.services.mock_service import MockAIInterviewService


def test_jd_analysis_schema_accepts_expected_shape():
    analysis = JDAnalysis(
        role_summary="Backend AI role focused on production LLM systems.",
        required_technical_skills=[
            SkillItem(name="Python", importance="must_have", evidence="Python listed as required.")
        ],
        preferred_skills=[],
        likely_interview_topics=["API design"],
        backend_system_design_topics=["Queues"],
        ai_llm_topics=["Prompting"],
        seniority_signals=["Own services end to end"],
        red_flags_or_ambiguities=[],
    )

    assert analysis.required_technical_skills[0].name == "Python"


def test_generate_questions_request_defaults_to_jd_language_matching():
    request = GenerateQuestionsRequest(
        resume_text="Python backend engineer with FastAPI, SQL, APIs, and AI workflow project experience.",
        job_description="We need a backend AI engineer with Python, APIs, and LLM application experience.",
        role_type="AI Engineer",
    )

    assert request.output_language == "Match job description language"
    assert request.demo_mode is False


def test_mock_service_generates_demo_question_set():
    service = MockAIInterviewService()
    questions = service.generate_questions(
        GenerateQuestionsRequest(
            resume_text="Python backend engineer with FastAPI, SQL, APIs, and AI workflow project experience.",
            job_description="We need a backend AI engineer with Python, APIs, and LLM application experience.",
            role_type="AI Engineer",
            demo_mode=True,
        )
    )

    assert questions.technical_questions
    assert questions.behavioral_questions
