from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.report_export import (
    build_markdown_report,
    build_pdf_report,
    pdf_filename,
    report_filename,
)


def get_api_base_url() -> str:
    try:
        configured_url = st.secrets.get("API_BASE_URL")
    except StreamlitSecretNotFoundError:
        configured_url = None

    if not configured_url:
        configured_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    return str(configured_url).rstrip("/")


API_BASE_URL = get_api_base_url()
ROLE_TYPES = [
    "Backend Engineer",
    "AI Engineer",
    "Agent Engineer",
    "Full-stack AI Engineer",
]
OUTPUT_LANGUAGES = [
    "Match job description language",
    "English",
    "Chinese",
]
OUTPUT_LANGUAGE_LABELS = {
    "Match job description language": "Match JD",
    "English": "English",
    "Chinese": "Chinese",
}


st.set_page_config(page_title="AI Interview Copilot", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #ffffff;
        color: #242833;
    }

    .block-container { padding-top: 1.6rem; max-width: 1220px; }
    h1, h2, h3, p, li, label { letter-spacing: 0; }
    h1, h2, h3, p, li, label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #242833;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #4f5665;
        padding: 8px 12px;
    }
    .stTabs [aria-selected="true"] {
        color: #242833;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e6e8ec;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background: #f7f8fa;
        color: #242833;
    }
    section[data-testid="stSidebar"] * {
        color-scheme: light;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] > div,
    textarea {
        background-color: #ffffff;
        color: #242833;
        -webkit-text-fill-color: #242833;
    }
    div[data-baseweb="input"] input:disabled {
        background-color: #f4f6f8;
        color: #667085;
        -webkit-text-fill-color: #667085;
        opacity: 1;
    }
    .small-muted { color: #667085; font-size: 0.9rem; }
    .workflow-panel {
        margin-top: 2.6rem;
        max-width: 280px;
        opacity: 0.78;
    }
    .workflow-kicker {
        color: #a5abb7;
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }
    .workflow-title {
        color: #575d6d;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.85rem;
    }
    .pipeline-step {
        display: grid;
        grid-template-columns: 23px 1fr;
        column-gap: 0.55rem;
        align-items: start;
        margin-bottom: 0.72rem;
    }
    .step-index {
        align-items: center;
        background: #fafafa;
        border: 1px solid #eceef3;
        border-radius: 50%;
        color: #9aa1af;
        display: flex;
        font-size: 0.7rem;
        font-weight: 700;
        height: 22px;
        justify-content: center;
        line-height: 1;
        width: 22px;
    }
    .step-title {
        color: #5e6472;
        font-size: 0.84rem;
        font-weight: 650;
        line-height: 1.25;
        margin-bottom: 0.08rem;
    }
    .step-copy {
        color: #9aa1af;
        font-size: 0.76rem;
        line-height: 1.28;
    }
    .workflow-note {
        border-top: 1px solid #eceef3;
        color: #9aa1af;
        font-size: 0.74rem;
        line-height: 1.32;
        margin-top: 0.9rem;
        padding-top: 0.72rem;
    }
    div[data-testid="stDownloadButton"] > button {
        min-height: 2.05rem;
        padding: 0.25rem 0.55rem;
        font-size: 0.82rem;
        line-height: 1;
    }
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stButton"] button[kind="primary"] * {
        color: #ffffff;
        -webkit-text-fill-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def create_session(
    resume_file,
    job_description: str,
    role_type: str,
    output_language: str,
    demo_mode: bool,
) -> dict[str, Any]:
    files = {"resume_pdf": (resume_file.name, resume_file.getvalue(), "application/pdf")}
    data = {
        "job_description": job_description,
        "role_type": role_type,
        "output_language": output_language,
        "demo_mode": demo_mode,
    }
    response = requests.post(
        f"{API_BASE_URL}/sessions/from-upload",
        files=files,
        data=data,
        timeout=240,
    )
    response.raise_for_status()
    return response.json()


def friendly_api_error(exc: requests.HTTPError) -> tuple[str, str | None]:
    if exc.response is None:
        return str(exc), None
    try:
        payload = exc.response.json()
    except ValueError:
        return exc.response.text, None

    detail = payload.get("detail", payload)
    if isinstance(detail, dict):
        return detail.get("message", "The request failed."), detail.get("action")
    if isinstance(detail, str):
        return detail, None
    return str(detail), None


def show_skill_list(title: str, items: list[dict[str, Any]]) -> None:
    st.subheader(title)
    if not items:
        st.caption("No items returned.")
        return
    st.dataframe(
        [
            {
                "Skill": item.get("name", ""),
                "Importance": item.get("importance", ""),
                "Evidence": item.get("evidence", ""),
            }
            for item in items
        ],
        use_container_width=True,
        hide_index=True,
    )


def show_questions(title: str, items: list[dict[str, Any]]) -> None:
    st.subheader(title)
    for item in items:
        st.markdown(f"**{item.get('question', '')}**")
        st.caption(f"{item.get('difficulty', '').title()} | {item.get('why_it_matters', '')}")


def show_session(session: dict[str, Any]) -> None:
    jd_analysis = session["jd_analysis"]
    resume_match = session["resume_match"]
    questions = session["questions"]
    answers = session["answers"]["answers"]

    st.divider()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Fit score", resume_match["overall_fit_score"])
    metric_cols[1].metric("Missing skills", len(resume_match["missing_skills"]))
    metric_cols[2].metric("Questions", sum(len(v) for v in questions.values()))
    metric_cols[3].metric("Answers", len(answers))

    meta_col, md_col, pdf_col = st.columns([1, 0.12, 0.13], vertical_alignment="center")
    with meta_col:
        st.caption(f"Output language: {session.get('output_language', 'Match job description language')}")
    with md_col:
        st.download_button(
            "↓ MD",
            data=build_markdown_report(session),
            file_name=report_filename(session),
            mime="text/markdown",
            use_container_width=True,
            help="Download Markdown report",
        )
    with pdf_col:
        st.download_button(
            "↓ PDF",
            data=build_pdf_report(session),
            file_name=pdf_filename(session),
            mime="application/pdf",
            use_container_width=True,
            help="Download PDF report",
        )

    if session.get("demo_mode"):
        st.info("Demo mode result: sample data was generated without OpenAI API calls.")

    tabs = st.tabs(["Overview", "JD Analysis", "Resume Match", "Questions", "Answers", "Raw JSON"])

    with tabs[0]:
        st.subheader("Role summary")
        st.write(jd_analysis["role_summary"])
        st.subheader("Positioning strategy")
        st.write(resume_match["suggested_positioning_strategy"])
        st.subheader("Project talking points")
        for point in resume_match["recommended_project_talking_points"]:
            st.markdown(f"- {point}")

    with tabs[1]:
        show_skill_list("Required technical skills", jd_analysis["required_technical_skills"])
        show_skill_list("Preferred skills", jd_analysis["preferred_skills"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Likely interview topics")
            for item in jd_analysis["likely_interview_topics"]:
                st.markdown(f"- {item}")
            st.subheader("Backend/system design topics")
            for item in jd_analysis["backend_system_design_topics"]:
                st.markdown(f"- {item}")
        with col_b:
            st.subheader("AI/LLM topics")
            for item in jd_analysis["ai_llm_topics"]:
                st.markdown(f"- {item}")
            st.subheader("Ambiguities")
            for item in jd_analysis["red_flags_or_ambiguities"]:
                st.markdown(f"- {item}")

    with tabs[2]:
        st.subheader("Strong and partial matches")
        st.dataframe(resume_match["strong_matches"], use_container_width=True, hide_index=True)
        st.subheader("Missing skills")
        for item in resume_match["missing_skills"]:
            st.markdown(f"- {item}")
        st.subheader("Risks to prepare for")
        for item in resume_match["resume_risks_to_prepare_for"]:
            st.markdown(f"- {item}")

    with tabs[3]:
        show_questions("Technical", questions["technical_questions"])
        show_questions("Project deep-dive", questions["project_deep_dive_questions"])
        show_questions("System design", questions["system_design_questions"])
        show_questions("Behavioral", questions["behavioral_questions"])

    with tabs[4]:
        for answer in answers:
            with st.expander(f"{answer['category']}: {answer['question']}", expanded=False):
                st.write(answer["concise_answer"])
                st.caption("Resume evidence used")
                for evidence in answer["resume_evidence_used"]:
                    st.markdown(f"- {evidence}")
                st.caption(answer["honesty_guardrail"])

    with tabs[5]:
        st.json(session)


with st.sidebar:
    st.title("AI Interview Copilot")
    st.caption("Resume + JD -> interview strategy, questions, and grounded answers.")
    selected_role = st.selectbox("Target role type", ROLE_TYPES)
    selected_output_language = st.selectbox(
        "Output language",
        OUTPUT_LANGUAGES,
        format_func=lambda value: OUTPUT_LANGUAGE_LABELS[value],
    )
    demo_mode = st.toggle(
        "Demo mode",
        value=False,
        help="Use sample AI outputs without spending OpenAI API credits.",
    )
    st.text_input("Backend API", value=API_BASE_URL, disabled=True)

    st.divider()
    st.subheader("Saved sessions")
    try:
        summaries = api_get("/sessions")
        for summary in summaries:
            mode = " | demo" if summary.get("demo_mode") else ""
            label = f"#{summary['id']} | {summary['role_type']} | {summary['overall_fit_score']}{mode}"
            if st.button(label, key=f"session-{summary['id']}", use_container_width=True):
                st.session_state["active_session"] = api_get(f"/sessions/{summary['id']}")
    except requests.RequestException:
        st.caption("Start the FastAPI backend to load sessions.")


st.title("Interview prep workspace")
st.markdown(
    '<p class="small-muted">Upload a PDF resume, paste a target JD, and generate a saved preparation session.</p>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.42, 0.58], gap="large")

with left:
    resume_file = st.file_uploader("Resume PDF", type=["pdf"])
    job_description = st.text_area("Target job description", height=320)
    run = st.button("Generate prep session", type="primary", use_container_width=True)

with right:
    st.markdown(
        """
        <div class="workflow-panel">
            <div class="workflow-kicker">Run flow</div>
            <div class="workflow-title">Prep pipeline</div>
            <div class="pipeline-step">
                <div class="step-index">1</div>
                <div>
                    <div class="step-title">Resume intake</div>
                    <div class="step-copy">Extracts clean text from the uploaded PDF.</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-index">2</div>
                <div>
                    <div class="step-title">Role analysis</div>
                    <div class="step-copy">Maps JD requirements to skills, topics, and signals.</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-index">3</div>
                <div>
                    <div class="step-title">Interview pack</div>
                    <div class="step-copy">Builds matching insights, questions, and grounded answers.</div>
                </div>
            </div>
            <div class="pipeline-step">
                <div class="step-index">4</div>
                <div>
                    <div class="step-title">Saved session</div>
                    <div class="step-copy">Stores the generated prep package for review.</div>
                </div>
            </div>
            <div class="workflow-note">
                Uses the sidebar language setting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if run:
    if resume_file is None:
        st.error("Please upload a resume PDF.")
    elif len(job_description.strip()) < 50:
        st.error("Please paste a fuller job description.")
    else:
        with st.spinner("Running the interview copilot workflow..."):
            try:
                st.session_state["active_session"] = create_session(
                    resume_file,
                    job_description.strip(),
                    selected_role,
                    selected_output_language,
                    demo_mode,
                )
                if st.session_state["active_session"].get("demo_mode"):
                    st.success("Demo preview generated. It was not saved to session history.")
                else:
                    st.success("Session generated and saved.")
            except requests.HTTPError as exc:
                message, action = friendly_api_error(exc)
                st.error(message)
                if action:
                    st.info(action)
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")

if "active_session" in st.session_state:
    show_session(st.session_state["active_session"])
