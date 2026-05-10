from app.services.report_export import build_markdown_report, build_pdf_report, pdf_filename, report_filename


def test_build_markdown_report_includes_core_sections():
    session = {
        "id": 7,
        "created_at": "2026-05-10T00:00:00+00:00",
        "role_type": "AI Engineer",
        "output_language": "English",
        "demo_mode": False,
        "jd_analysis": {
            "role_summary": "Backend AI role.",
            "required_technical_skills": [
                {"name": "Python", "importance": "must_have", "evidence": "Listed in JD."}
            ],
            "preferred_skills": [],
            "likely_interview_topics": ["API design"],
            "backend_system_design_topics": ["Queues"],
            "ai_llm_topics": ["Structured outputs"],
            "seniority_signals": [],
            "red_flags_or_ambiguities": [],
        },
        "resume_match": {
            "overall_fit_score": 82,
            "strong_matches": [
                {"skill": "Python", "strength": "strong", "resume_evidence": "Resume project."}
            ],
            "missing_skills": ["Kubernetes"],
            "transferable_strengths": ["Backend APIs"],
            "suggested_positioning_strategy": "Lead with AI workflow delivery.",
            "recommended_project_talking_points": ["Interview Copilot"],
            "resume_risks_to_prepare_for": ["Deployment depth"],
        },
        "questions": {
            "technical_questions": [
                {"question": "How do you parse PDFs?", "why_it_matters": "Document processing.", "difficulty": "medium"}
            ],
            "project_deep_dive_questions": [],
            "system_design_questions": [],
            "behavioral_questions": [],
        },
        "answers": {
            "answers": [
                {
                    "category": "Technical",
                    "question": "How do you parse PDFs?",
                    "concise_answer": "I use pdfplumber with a PyPDF2 fallback.",
                    "resume_evidence_used": ["Interview Copilot project"],
                    "honesty_guardrail": "Do not invent OCR experience.",
                }
            ]
        },
    }

    report = build_markdown_report(session)

    assert "# AI Interview Prep Report" in report
    assert "## JD Analysis" in report
    assert "## Personalized Answers" in report
    assert "I use pdfplumber" in report
    assert report_filename(session) == "ai-interview-prep-ai-engineer-session-7.md"
    assert pdf_filename(session) == "ai-interview-prep-ai-engineer-session-7.pdf"

    pdf = build_pdf_report(session)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000
