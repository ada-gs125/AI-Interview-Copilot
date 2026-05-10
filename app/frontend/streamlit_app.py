import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
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
    .block-container { padding-top: 1.6rem; max-width: 1220px; }
    h1, h2, h3 { letter-spacing: 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: 8px 12px; }
    div[data-testid="stMetric"] {
        border: 1px solid #e6e8ec;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
    }
    section[data-testid="stSidebar"] { background: #f7f8fa; }
    .small-muted { color: #667085; font-size: 0.9rem; }
    .workflow-panel {
        margin-top: 1.9rem;
        max-width: 440px;
    }
    .workflow-kicker {
        color: #ef5350;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }
    .workflow-title {
        color: #2f3040;
        font-size: 1.28rem;
        font-weight: 750;
        margin-bottom: 1rem;
    }
    .pipeline-step {
        display: grid;
        grid-template-columns: 34px 1fr;
        column-gap: 0.8rem;
        align-items: start;
        margin-bottom: 1rem;
    }
    .step-index {
        align-items: center;
        background: #fff3f2;
        border: 1px solid #ffd8d5;
        border-radius: 50%;
        color: #d93d38;
        display: flex;
        font-size: 0.82rem;
        font-weight: 750;
        height: 30px;
        justify-content: center;
        line-height: 1;
        width: 30px;
    }
    .step-title {
        color: #303141;
        font-size: 0.98rem;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 0.18rem;
    }
    .step-copy {
        color: #7b8190;
        font-size: 0.88rem;
        line-height: 1.35;
    }
    .workflow-note {
        border-top: 1px solid #eceef3;
        color: #7b8190;
        font-size: 0.86rem;
        line-height: 1.4;
        margin-top: 1.15rem;
        padding-top: 0.95rem;
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
) -> dict[str, Any]:
    files = {"resume_pdf": (resume_file.name, resume_file.getvalue(), "application/pdf")}
    data = {
        "job_description": job_description,
        "role_type": role_type,
        "output_language": output_language,
    }
    response = requests.post(
        f"{API_BASE_URL}/sessions/from-upload",
        files=files,
        data=data,
        timeout=240,
    )
    response.raise_for_status()
    return response.json()


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
    st.caption(f"Output language: {session.get('output_language', 'Match job description language')}")

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
    st.text_input("Backend API", value=API_BASE_URL, disabled=True)

    st.divider()
    st.subheader("Saved sessions")
    try:
        summaries = api_get("/sessions")
        for summary in summaries:
            label = f"#{summary['id']} | {summary['role_type']} | {summary['overall_fit_score']}"
            if st.button(label, key=f"session-{summary['id']}", use_container_width=True):
                st.session_state["active_session"] = api_get(f"/sessions/{summary['id']}")
    except requests.RequestException:
        st.caption("Start the FastAPI backend to load sessions.")


st.title("Interview prep workspace")
st.markdown(
    '<p class="small-muted">Upload a PDF resume, paste a target JD, and generate a saved preparation session.</p>',
    unsafe_allow_html=True,
)

left, right = st.columns([0.9, 1.1], gap="large")

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
                Output language follows the sidebar setting, with Match JD using the job description language.
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
                )
                st.success("Session generated and saved.")
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"API error: {detail}")
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")

if "active_session" in st.session_state:
    show_session(st.session_state["active_session"])
