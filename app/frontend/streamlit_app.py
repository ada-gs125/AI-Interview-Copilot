from __future__ import annotations

import os
import sys
import time
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
from app.frontend.api_client import (
    api_delete,
    api_get,
    create_session_from_upload,
    create_session_job,
    friendly_api_error,
    get_session_job,
    login_user,
    register_user,
    should_fallback_to_sync,
)


DEFAULT_API_BASE_URL = "https://backend-production-b0243.up.railway.app"


def get_api_base_url() -> str:
    try:
        configured_url = st.secrets.get("API_BASE_URL")
    except StreamlitSecretNotFoundError:
        configured_url = None

    if not configured_url:
        configured_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)

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
TERMINAL_JOB_STATUSES = {"succeeded", "failed"}
STEP_LABELS = {
    "parse_resume": "Resume intake",
    "analyze_jd": "JD analysis",
    "match_resume": "Resume match",
    "generate_questions": "Question generation",
    "generate_answers": "Answer generation",
    "save_session": "Session save",
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

    /* ── Auth page ── */
    .auth-brand {
        text-align: center;
        padding: 1.6rem 0 1.2rem;
    }
    .auth-brand-title {
        font-size: 1.45rem;
        font-weight: 720;
        color: #242833;
        letter-spacing: -0.02em;
        margin: 0 0 0.3rem;
    }
    .auth-brand-sub {
        color: #8b93a4;
        font-size: 0.84rem;
        margin: 0;
        letter-spacing: 0.01em;
    }

    /* ── Sidebar user badge ── */
    .user-badge {
        background: #eef0f5;
        border-radius: 8px;
        padding: 10px 13px;
        margin: 6px 0 10px;
    }
    .user-badge-label {
        font-size: 0.67rem;
        color: #9aa1af;
        font-weight: 680;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        display: block;
        margin-bottom: 3px;
    }
    .user-badge-email {
        font-size: 0.83rem;
        color: #3d4351;
        word-break: break-all;
    }

    /* ── Session list items ── */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        border: 1px solid #e2e5eb;
        border-radius: 7px;
        background: #ffffff;
        text-align: left;
        font-size: 0.8rem;
        color: #3d4351;
        padding: 7px 10px;
        margin-bottom: 2px;
        transition: background 0.15s, border-color 0.15s;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: #f4f5f9;
        border-color: #c8cdd8;
    }

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


def show_job_progress(job: dict[str, Any]) -> None:
    status = job.get("status", "queued")
    progress = int(job.get("progress_percent", 0))
    current_step = job.get("current_step") or "queued"
    job_id = job.get("id", "")

    top_cols = st.columns([0.42, 0.22, 0.18, 0.18])
    top_cols[0].metric("Job", job_id[:8] if job_id else "pending")
    top_cols[1].metric("Status", status)
    top_cols[2].metric("Progress", f"{progress}%")
    top_cols[3].metric("Step", STEP_LABELS.get(current_step, current_step))
    st.progress(progress)

    steps = job.get("steps", [])
    if steps:
        st.dataframe(
            [
                {
                    "Step": STEP_LABELS.get(step.get("name", ""), step.get("name", "")),
                    "Status": step.get("status", ""),
                    "Latency ms": step.get("latency_ms") or "",
                    "Calls": step.get("usage", {}).get("call_count", 0),
                    "Tokens": step.get("usage", {}).get("total_tokens", ""),
                    "Cost USD": step.get("usage", {}).get("estimated_cost_usd", ""),
                }
                for step in steps
            ],
            use_container_width=True,
            hide_index=True,
        )

    usage = job.get("usage", {})
    usage_bits = []
    if usage.get("call_count") is not None:
        usage_bits.append(f"AI calls: {usage.get('call_count', 0)}")
    if usage.get("total_tokens"):
        usage_bits.append(f"Tokens: {usage['total_tokens']}")
    if usage.get("estimated_cost_usd"):
        usage_bits.append(f"Estimated cost: ${usage['estimated_cost_usd']}")
    if usage_bits:
        st.caption(" | ".join(usage_bits))


def poll_session_job(status_url: str, access_token: str) -> dict[str, Any]:
    placeholder = st.empty()
    job = get_session_job(API_BASE_URL, status_url, access_token)
    with placeholder.container():
        show_job_progress(job)

    while job.get("status") not in TERMINAL_JOB_STATUSES:
        time.sleep(1.25)
        job = get_session_job(API_BASE_URL, status_url, access_token)
        placeholder.empty()
        with placeholder.container():
            show_job_progress(job)

    return job


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

    if session.get("id"):
        if st.button("Delete session", use_container_width=False):
            try:
                api_delete(API_BASE_URL, f"/sessions/{session['id']}", st.session_state.get("access_token"))
                st.session_state.pop("active_session", None)
                st.success("Session deleted.")
                st.rerun()
            except requests.HTTPError as exc:
                message, action = friendly_api_error(exc)
                st.error(message)
                if action:
                    st.info(action)
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")

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


def _handle_auth_error(exc: Exception) -> None:
    if isinstance(exc, requests.HTTPError):
        message, action = friendly_api_error(exc)
        st.error(message)
        if action:
            st.info(action)
    else:
        st.error(f"Could not reach backend: {exc}")


# Read auth state once — used in both sidebar and main area
access_token = st.session_state.get("access_token")
current_user_data = st.session_state.get("current_user")

with st.sidebar:
    st.title("AI Interview Copilot")
    st.caption("Resume + JD -> interview strategy, questions, and grounded answers.")

    if access_token and current_user_data:
        st.markdown(
            f"""
            <div class="user-badge">
                <span class="user-badge-label">Signed in as</span>
                <span class="user-badge-email">{current_user_data.get("email", "")}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Sign out", use_container_width=True):
            for key in ("access_token", "current_user", "active_session", "active_job"):
                st.session_state.pop(key, None)
            st.rerun()

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

        st.divider()
        st.subheader("Saved sessions")
        ROLE_ICONS = {
            "Backend Engineer": "⚙️",
            "AI Engineer": "🤖",
            "Agent Engineer": "🧠",
            "Full-stack AI Engineer": "🔷",
        }
        try:
            summaries = api_get(API_BASE_URL, "/sessions", access_token)
            for summary in summaries:
                icon = ROLE_ICONS.get(summary["role_type"], "📄")
                score = summary["overall_fit_score"]
                demo_tag = " · demo" if summary.get("demo_mode") else ""
                label = f"{icon} {summary['role_type']}  ·  {score}/100{demo_tag}"
                if st.button(label, key=f"session-{summary['id']}", use_container_width=True):
                    st.session_state["active_session"] = api_get(
                        API_BASE_URL,
                        f"/sessions/{summary['id']}",
                        access_token,
                    )
        except requests.RequestException:
            st.caption("Start the FastAPI backend to load sessions.")


st.markdown(
    """
    <h1 style="font-size:2rem; font-weight:760; letter-spacing:-0.02em; margin-bottom:0.15rem;">
        Interview prep workspace
    </h1>
    <p class="small-muted" style="margin-top:0;">
        Upload a PDF resume, paste a target JD, and generate a saved preparation session.
    </p>
    """,
    unsafe_allow_html=True,
)

if not access_token:
    _, center, _ = st.columns([0.8, 1.8, 0.8])
    with center:
        st.markdown(
            """
            <div class="auth-brand">
                <div class="auth-brand-title">AI Interview Copilot</div>
                <p class="auth-brand-sub">Resume &middot; Job description &middot; Personalised prep kit</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sign_in_tab, create_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("login_form"):
                login_email = st.text_input("Email")
                login_password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", use_container_width=True, type="primary"):
                    if not login_email or not login_password:
                        st.error("Email and password are required.")
                    else:
                        try:
                            auth = login_user(API_BASE_URL, login_email, login_password)
                            st.session_state["access_token"] = auth["access_token"]
                            st.session_state["current_user"] = auth["user"]
                            st.rerun()
                        except Exception as exc:
                            _handle_auth_error(exc)

        with create_tab:
            with st.form("register_form"):
                reg_email = st.text_input("Email")
                reg_password = st.text_input("Password", type="password", help="Minimum 8 characters")
                reg_confirm = st.text_input("Confirm password", type="password")
                if st.form_submit_button("Create account", use_container_width=True, type="primary"):
                    if not reg_email or not reg_password:
                        st.error("Email and password are required.")
                    elif len(reg_password) < 8:
                        st.error("Password must be at least 8 characters.")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            auth = register_user(API_BASE_URL, reg_email, reg_password)
                            st.session_state["access_token"] = auth["access_token"]
                            st.session_state["current_user"] = auth["user"]
                            st.rerun()
                        except Exception as exc:
                            _handle_auth_error(exc)

    st.stop()

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
        st.session_state.pop("active_session", None)
        st.session_state.pop("active_job", None)
        with st.status("Starting background workflow...", expanded=True) as job_status:
            try:
                try:
                    job_create = create_session_job(
                        API_BASE_URL,
                        resume_file,
                        job_description.strip(),
                        selected_role,
                        selected_output_language,
                        demo_mode,
                        access_token,
                    )
                except requests.HTTPError as exc:
                    if not should_fallback_to_sync(exc):
                        raise
                    job_status.update(
                        label="Backend job API is unavailable. Using compatibility mode...",
                        state="running",
                    )
                    session = create_session_from_upload(
                        API_BASE_URL,
                        resume_file,
                        job_description.strip(),
                        selected_role,
                        selected_output_language,
                        demo_mode,
                        access_token,
                    )
                    st.session_state["active_session"] = session
                    job_status.update(label="Compatibility workflow completed.", state="complete")
                    if session.get("demo_mode"):
                        st.success("Demo preview generated. It was not saved to session history.")
                    else:
                        st.success("Session generated and saved.")
                else:
                    job_status.update(label=f"Queued job {job_create['job_id'][:8]}", state="running")
                    job = poll_session_job(job_create["status_url"], access_token)
                    st.session_state["active_job"] = job

                    if job.get("status") == "succeeded" and job.get("result"):
                        st.session_state["active_session"] = job["result"]
                        job_status.update(label="Background workflow completed.", state="complete")
                        if job["result"].get("demo_mode"):
                            st.success("Demo preview generated by background job. It was not saved to session history.")
                        else:
                            st.success("Session generated and saved by background job.")
                    else:
                        error = job.get("error") or {}
                        job_status.update(label="Background workflow failed.", state="error")
                        st.error(error.get("message", "The background workflow failed."))
                        if error.get("action"):
                            st.info(error["action"])
            except requests.HTTPError as exc:
                job_status.update(label="Request failed.", state="error")
                message, action = friendly_api_error(exc)
                st.error(message)
                if action:
                    st.info(action)
            except requests.RequestException as exc:
                job_status.update(label="Could not reach backend.", state="error")
                st.error(f"Could not reach backend: {exc}")

if "active_session" in st.session_state:
    show_session(st.session_state["active_session"])
