from app.schemas import GenerateQuestionsRequest, JDAnalysis, ResumeMatchRequest, SkillItem
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


def test_demo_preview_session_can_use_unsaved_id():
    from app.schemas import AnswerSet, SessionResponse

    service = MockAIInterviewService()
    jd = service.analyze_jd(
        "We need a backend AI engineer with Python, APIs, LLM workflows, persistence, and system design experience.",
        "AI Engineer",
    )
    match = service.match_resume(
        ResumeMatchRequest(
            resume_text="Python FastAPI backend engineer with APIs, SQL, and AI workflow project experience.",
            job_description="We need a backend AI engineer with Python, APIs, LLM workflows, persistence, and system design experience.",
            role_type="AI Engineer",
            demo_mode=True,
        )
    )
    questions = service.generate_questions(
        GenerateQuestionsRequest(
            resume_text="Python FastAPI backend engineer with APIs, SQL, and AI workflow project experience.",
            job_description="We need a backend AI engineer with Python, APIs, LLM workflows, persistence, and system design experience.",
            role_type="AI Engineer",
            demo_mode=True,
        )
    )

    session = SessionResponse(
        id=0,
        created_at="2026-05-10T00:00:00+00:00",
        role_type="AI Engineer",
        output_language="English",
        demo_mode=True,
        job_description="We need a backend AI engineer with Python, APIs, LLM workflows, persistence, and system design experience.",
        resume_text="Python FastAPI backend engineer with APIs, SQL, and AI workflow project experience.",
        jd_analysis=jd,
        resume_match=match,
        questions=questions,
        answers=AnswerSet(answers=[]),
    )

    assert session.id == 0
    assert session.demo_mode is True


def test_mock_batch_answers_use_chinese_categories_when_matching_chinese_jd():
    service = MockAIInterviewService()
    chinese_jd = (
        "我们需要一名 AI 工程师，熟悉 Python、后端 API、LLM 应用、系统设计、数据库持久化、"
        "结构化输出、提示词工程，并且能够独立交付端到端的 AI 产品功能。"
    )
    questions = service.generate_questions(
        GenerateQuestionsRequest(
            resume_text="Python FastAPI backend engineer with APIs, SQL, and AI workflow project experience.",
            job_description=chinese_jd,
            role_type="AI Engineer",
            output_language="Match job description language",
            demo_mode=True,
        )
    )

    answers = service.generate_answers_for_question_set(
        resume_text="Python FastAPI backend engineer with APIs, SQL, and AI workflow project experience.",
        job_description=chinese_jd,
        role_type="AI Engineer",
        output_language="Match job description language",
        questions=questions,
    )

    categories = {answer.category for answer in answers.answers}
    assert "技术问题" in categories
    assert "Technical" not in categories
