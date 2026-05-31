"""Markdown and PDF report builders for generated interview sessions."""

from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _bullet_list(items: list[str]) -> str:
    # Keep empty sections explicit in exported reports.
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def _question_section(title: str, items: list[dict[str, Any]]) -> str:
    lines = [f"## {title}"]
    if not items:
        lines.append("No questions generated.")
        return "\n\n".join(lines)

    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                f"### {index}. {item.get('question', '')}",
                f"- Difficulty: {item.get('difficulty', '')}",
                f"- Why it matters: {item.get('why_it_matters', '')}",
            ]
        )
    return "\n\n".join(lines)


def build_markdown_report(session: dict[str, Any]) -> str:
    # Frontend download path uses this as the source format and PDF input.
    jd_analysis = session["jd_analysis"]
    resume_match = session["resume_match"]
    questions = session["questions"]
    answers = session["answers"]["answers"]
    mode = "Demo preview" if session.get("demo_mode") else "Saved session"

    required_skills = [
        f"{item.get('name', '')} ({item.get('importance', '')}): {item.get('evidence', '')}"
        for item in jd_analysis["required_technical_skills"]
    ]
    preferred_skills = [
        f"{item.get('name', '')} ({item.get('importance', '')}): {item.get('evidence', '')}"
        for item in jd_analysis["preferred_skills"]
    ]
    strong_matches = [
        f"{item.get('skill', '')} [{item.get('strength', '')}]: {item.get('resume_evidence', '')}"
        for item in resume_match["strong_matches"]
    ]

    answer_lines = ["## Personalized Answers"]
    if not answers:
        answer_lines.append("No answers generated.")
    for index, answer in enumerate(answers, start=1):
        answer_lines.extend(
            [
                f"### {index}. {answer.get('question', '')}",
                f"- Category: {answer.get('category', '')}",
                "",
                answer.get("concise_answer", ""),
                "",
                "Evidence used:",
                _bullet_list(answer.get("resume_evidence_used", [])),
                "",
                f"Guardrail: {answer.get('honesty_guardrail', '')}",
            ]
        )

    sections = [
        "# AI Interview Prep Report",
        f"- Mode: {mode}",
        f"- Role type: {session.get('role_type', '')}",
        f"- Output language: {session.get('output_language', '')}",
        f"- Created at: {session.get('created_at', '')}",
        "",
        "## Role Summary",
        jd_analysis["role_summary"],
        "",
        "## Fit Overview",
        f"- Fit score: {resume_match['overall_fit_score']}/100",
        f"- Missing skill count: {len(resume_match['missing_skills'])}",
        "",
        "## JD Analysis",
        "### Required Technical Skills",
        _bullet_list(required_skills),
        "",
        "### Preferred Skills",
        _bullet_list(preferred_skills),
        "",
        "### Likely Interview Topics",
        _bullet_list(jd_analysis["likely_interview_topics"]),
        "",
        "### Backend/System Design Topics",
        _bullet_list(jd_analysis["backend_system_design_topics"]),
        "",
        "### AI/LLM Topics",
        _bullet_list(jd_analysis["ai_llm_topics"]),
        "",
        "## Resume Match",
        "### Strong Matches",
        _bullet_list(strong_matches),
        "",
        "### Missing Skills",
        _bullet_list(resume_match["missing_skills"]),
        "",
        "### Transferable Strengths",
        _bullet_list(resume_match["transferable_strengths"]),
        "",
        "### Positioning Strategy",
        resume_match["suggested_positioning_strategy"],
        "",
        "### Project Talking Points",
        _bullet_list(resume_match["recommended_project_talking_points"]),
        "",
        "### Risks To Prepare For",
        _bullet_list(resume_match["resume_risks_to_prepare_for"]),
        "",
        _question_section("Technical Questions", questions["technical_questions"]),
        "",
        _question_section("Project Deep-Dive Questions", questions["project_deep_dive_questions"]),
        "",
        _question_section("System Design Questions", questions["system_design_questions"]),
        "",
        _question_section("Behavioral Questions", questions["behavioral_questions"]),
        "",
        "\n\n".join(answer_lines),
    ]

    return "\n".join(sections).strip() + "\n"


def report_filename(session: dict[str, Any]) -> str:
    role = session.get("role_type", "interview").lower().replace(" ", "-")
    suffix = "demo" if session.get("demo_mode") else f"session-{session.get('id', 'report')}"
    return f"ai-interview-prep-{role}-{suffix}.md"


def pdf_filename(session: dict[str, Any]) -> str:
    return report_filename(session).replace(".md", ".pdf")


def build_pdf_report(session: dict[str, Any]) -> bytes:
    # Render the markdown-like report into a simple ReportLab document.
    markdown = build_markdown_report(session)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="AI Interview Prep Report",
    )

    font_name = _register_report_font()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=20,
            leading=25,
            spaceAfter=14,
            textColor=colors.HexColor("#303141"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=17,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#303141"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading3"],
            fontName=font_name,
            fontSize=10.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#4f5665"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.2,
            leading=13,
            spaceAfter=5,
            textColor=colors.HexColor("#3b3f4a"),
        )
    )

    story = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 0.08 * inch))
            continue

        if line.startswith("# "):
            story.append(Paragraph(_escape(line[2:]), styles["ReportTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(_escape(line[3:]), styles["SectionHeading"]))
        elif line.startswith("### "):
            story.append(Paragraph(_escape(line[4:]), styles["SubHeading"]))
        elif line.startswith("- "):
            story.append(Paragraph(f"&#8226; {_escape(line[2:])}", styles["Body"]))
        else:
            story.append(Paragraph(_escape(line), styles["Body"]))

    doc.build(story)
    return buffer.getvalue()


def _register_report_font() -> str:
    # Prefer CJK-capable fonts so Chinese reports render correctly.
    for path in (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("ReportFont", path))
                return "ReportFont"
            except Exception:
                continue
    return "Helvetica"


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
