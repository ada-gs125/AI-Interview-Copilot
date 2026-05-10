from app.schemas import JDAnalysis, SkillItem


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

