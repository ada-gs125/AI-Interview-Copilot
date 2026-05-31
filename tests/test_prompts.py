from app.services.prompts import (
    PROMPT_VERSION,
    generate_answer_prompt,
    generate_questions_prompt,
    jd_analysis_prompt,
)
from app.services.llm_skills import JD_ANALYSIS_SKILL, all_skill_specs

from tests.factories import ENGLISH_JD, RESUME_TEXT, sample_jd_analysis, sample_resume_match


def test_prompt_version_is_defined():
    assert PROMPT_VERSION


def test_llm_skill_metadata_wraps_prompt_builder():
    prompt = JD_ANALYSIS_SKILL.prompt(
        job_description=ENGLISH_JD,
        role_type="AI Engineer",
        output_language="English",
    )
    specs = {spec.name: spec for spec in all_skill_specs()}

    assert JD_ANALYSIS_SKILL.name == "jd_analysis"
    assert JD_ANALYSIS_SKILL.version == PROMPT_VERSION
    assert JD_ANALYSIS_SKILL.output_schema.__name__ == "JDAnalysis"
    assert "jd_analysis" in specs
    assert ENGLISH_JD in prompt.user


def test_jd_prompt_contains_faithfulness_and_language_rules():
    prompt = jd_analysis_prompt(
        job_description=ENGLISH_JD,
        role_type="AI Engineer",
        output_language="English",
    )

    assert "job description only" in prompt.system
    assert "do not invent" in prompt.system
    assert "Write every user-facing field in English" in prompt.system
    assert ENGLISH_JD in prompt.user


def test_question_prompt_includes_context_and_few_shot_examples():
    prompt = generate_questions_prompt(
        resume_text=RESUME_TEXT,
        job_description=ENGLISH_JD,
        role_type="AI Engineer",
        output_language="English",
        jd_context=sample_jd_analysis().model_dump_json(),
        match_context=sample_resume_match().model_dump_json(),
        few_shot_examples=[{"question": "How do you design APIs?", "answer": "Use FastAPI."}],
    )

    assert "must not imply experience absent from the resume" in prompt.system
    assert "High-relevance questions" in prompt.user
    assert "How do you design APIs?" in prompt.user


def test_answer_prompt_requires_traceable_resume_evidence_for_structured_output():
    prompt = generate_answer_prompt(
        resume_text=RESUME_TEXT,
        job_description=ENGLISH_JD,
        role_type="AI Engineer",
        output_language="English",
        question="How do you make LLM output reliable enough for app code?",
        category="Technical",
        structured=True,
    )

    assert "resume_evidence_used" in prompt.system
    assert "traceable to the resume" in prompt.system
    assert RESUME_TEXT in prompt.user
