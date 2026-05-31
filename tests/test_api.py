import json

import pytest
from fastapi.testclient import TestClient

from app.schemas import SessionJobResponse, SessionSummary, UserResponse
from tests.factories import CHINESE_JD, ENGLISH_JD, RESUME_TEXT, sample_session


def _client_without_startup_db(monkeypatch, *, authenticated: bool = True, user_id: int = 1):
    import app.main as main
    import app.routes.interview as interview_routes

    monkeypatch.setattr(main, "init_db", lambda: None)
    if authenticated:
        main.app.dependency_overrides[interview_routes.current_user] = lambda: UserResponse(
            id=user_id,
            email=f"user{user_id}@example.com",
            created_at="2026-05-11T00:00:00+00:00",
        )
    else:
        main.app.dependency_overrides.pop(interview_routes.current_user, None)
    return TestClient(main.app)


def test_health_endpoint(monkeypatch):
    with _client_without_startup_db(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_and_login_return_access_tokens(monkeypatch):
    import app.routes.auth as auth_routes

    user = UserResponse(id=7, email="user@example.com", created_at="2026-05-11T00:00:00+00:00")
    monkeypatch.setattr(auth_routes, "create_user", lambda *, email, password: user)
    monkeypatch.setattr(auth_routes, "authenticate_user", lambda *, email, password: user)

    with _client_without_startup_db(monkeypatch, authenticated=False) as client:
        register_response = client.post(
            "/auth/register",
            json={"email": "user@example.com", "password": "password123"},
        )
        login_response = client.post(
            "/auth/login",
            json={"email": "user@example.com", "password": "password123"},
        )

    assert register_response.status_code == 201
    assert register_response.json()["access_token"]
    assert register_response.json()["user"]["email"] == "user@example.com"
    assert login_response.status_code == 200
    assert login_response.json()["access_token"]


def test_login_rejects_bad_credentials(monkeypatch):
    import app.routes.auth as auth_routes

    monkeypatch.setattr(auth_routes, "authenticate_user", lambda *, email, password: None)

    with _client_without_startup_db(monkeypatch, authenticated=False) as client:
        response = client.post(
            "/auth/login",
            json={"email": "user@example.com", "password": "password123"},
        )

    assert response.status_code == 401


def test_sessions_require_authentication(monkeypatch):
    with _client_without_startup_db(monkeypatch, authenticated=False) as client:
        response = client.get("/sessions")

    assert response.status_code == 401


def test_generate_answer_stream_returns_sse_tokens(monkeypatch):
    with _client_without_startup_db(monkeypatch) as client:
        response = client.post(
            "/generate-answer/stream",
            json={
                "resume_text": RESUME_TEXT,
                "job_description": ENGLISH_JD,
                "role_type": "AI Engineer",
                "output_language": "English",
                "demo_mode": True,
                "question": "How do you make LLM output reliable for application code?",
                "category": "Technical",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    tokens = [json.loads(line[6:]) for line in lines[:-1]]
    assert "".join(tokens)


@pytest.fixture
def fake_session_job_store(monkeypatch):
    import app.routes.interview as interview_routes

    jobs = {}

    def fake_create(*, job_id, user_id, role_type, output_language, demo_mode):
        jobs[job_id] = {
            "id": job_id,
            "user_id": user_id,
            "status": "queued",
            "created_at": "2026-05-11T00:00:00+00:00",
            "updated_at": "2026-05-11T00:00:00+00:00",
            "completed_at": None,
            "current_step": None,
            "progress_percent": 0,
            "role_type": role_type,
            "output_language": output_language,
            "demo_mode": demo_mode,
            "session_id": None,
            "error": None,
            "steps": [],
            "usage": {},
            "result": None,
        }
        return job_id

    def fake_update(job_id, **kwargs):
        job = jobs[job_id]
        job.update({key: value for key, value in kwargs.items() if value is not None})
        if kwargs.get("completed"):
            job["completed_at"] = "2026-05-11T00:00:01+00:00"
        job["updated_at"] = "2026-05-11T00:00:01+00:00"

    def fake_get(job_id, *, user_id):
        job = jobs.get(job_id)
        if job is None or job["user_id"] != user_id:
            return None
        return SessionJobResponse.model_validate(job)

    monkeypatch.setattr(interview_routes, "create_session_job", fake_create)
    monkeypatch.setattr(interview_routes, "update_session_job", fake_update)
    monkeypatch.setattr(interview_routes, "get_session_job", fake_get)

    return jobs


def test_create_session_job_runs_background_workflow_and_returns_status(monkeypatch, fake_session_job_store):
    import app.routes.interview as interview_routes

    monkeypatch.setattr(interview_routes, "extract_resume_text", lambda _: RESUME_TEXT)

    with _client_without_startup_db(monkeypatch) as client:
        create_response = client.post(
            "/sessions/jobs",
            data={
                "job_description": CHINESE_JD,
                "role_type": "AI Engineer",
                "output_language": "Match job description language",
                "demo_mode": "true",
            },
            files={"resume_pdf": ("resume.pdf", b"%PDF demo", "application/pdf")},
        )
        job_id = create_response.json()["job_id"]
        detail_response = client.get(f"/sessions/jobs/{job_id}")

    assert create_response.status_code == 202
    assert create_response.json()["status_url"] == f"/sessions/jobs/{job_id}"
    payload = detail_response.json()
    assert payload["status"] == "succeeded"
    assert payload["progress_percent"] == 100
    assert payload["current_step"] == "completed"
    assert [step["name"] for step in payload["steps"]] == [
        "parse_resume",
        "analyze_jd",
        "match_resume",
        "generate_questions",
        "generate_answers",
        "save_session",
    ]
    assert payload["result"]["demo_mode"] is True


def test_create_session_job_records_pdf_parse_failure(monkeypatch, fake_session_job_store):
    with _client_without_startup_db(monkeypatch) as client:
        create_response = client.post(
            "/sessions/jobs",
            data={
                "job_description": ENGLISH_JD,
                "role_type": "AI Engineer",
                "output_language": "English",
                "demo_mode": "true",
            },
            files={"resume_pdf": ("resume.pdf", b"not a pdf", "application/pdf")},
        )
        job_id = create_response.json()["job_id"]
        detail_response = client.get(f"/sessions/jobs/{job_id}")

    assert create_response.status_code == 202
    payload = detail_response.json()
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "pdf_parse_error"
    assert payload["steps"][0]["name"] == "parse_resume"
    assert payload["steps"][0]["status"] == "failed"


def test_session_job_detail_returns_404_for_missing_job(monkeypatch):
    import app.routes.interview as interview_routes

    monkeypatch.setattr(interview_routes, "get_session_job", lambda job_id, *, user_id: None)
    with _client_without_startup_db(monkeypatch) as client:
        response = client.get("/sessions/jobs/missing")

    assert response.status_code == 404


def test_openapi_exposes_session_job_endpoints(monkeypatch):
    with _client_without_startup_db(monkeypatch) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/sessions/jobs" in paths
    assert "/sessions/jobs/{job_id}" in paths


def test_sessions_endpoints_return_summaries_and_detail(monkeypatch):
    import app.routes.interview as interview_routes

    session = sample_session(session_id=42)
    summary = SessionSummary(
        id=session.id,
        created_at=session.created_at,
        role_type=session.role_type,
        output_language=session.output_language,
        demo_mode=session.demo_mode,
        overall_fit_score=session.resume_match.overall_fit_score,
        role_summary=session.jd_analysis.role_summary,
        missing_skill_count=len(session.resume_match.missing_skills),
    )
    monkeypatch.setattr(interview_routes, "delete_expired_sessions", lambda *, user_id, retention_days: 0)
    monkeypatch.setattr(interview_routes, "list_sessions", lambda *, user_id: [summary] if user_id == 1 else [])
    monkeypatch.setattr(
        interview_routes,
        "get_session",
        lambda session_id, *, user_id: session if session_id == 42 and user_id == 1 else None,
    )

    with _client_without_startup_db(monkeypatch) as client:
        list_response = client.get("/sessions")
        detail_response = client.get("/sessions/42")
        missing_response = client.get("/sessions/999")

    assert list_response.status_code == 200
    assert list_response.json()[0]["id"] == 42
    assert detail_response.status_code == 200
    assert detail_response.json()["job_description"] == ENGLISH_JD
    assert missing_response.status_code == 404


def test_delete_session_is_scoped_to_current_user(monkeypatch):
    import app.routes.interview as interview_routes

    deleted = []

    def fake_delete_session(session_id, *, user_id):
        deleted.append((session_id, user_id))
        return session_id == 42 and user_id == 1

    monkeypatch.setattr(interview_routes, "delete_session", fake_delete_session)

    with _client_without_startup_db(monkeypatch, user_id=1) as client:
        ok_response = client.delete("/sessions/42")
        missing_response = client.delete("/sessions/43")

    assert ok_response.status_code == 204
    assert missing_response.status_code == 404
    assert deleted == [(42, 1), (43, 1)]


def test_auth_me_returns_current_user(monkeypatch):
    with _client_without_startup_db(monkeypatch, user_id=5) as client:
        response = client.get("/auth/me")

    assert response.status_code == 200
    assert response.json()["id"] == 5
    assert response.json()["email"] == "user5@example.com"


def test_register_rejects_duplicate_email(monkeypatch):
    import app.routes.auth as auth_routes

    def raise_duplicate(*, email, password):
        raise ValueError("A user with this email already exists.")

    monkeypatch.setattr(auth_routes, "create_user", raise_duplicate)

    with _client_without_startup_db(monkeypatch, authenticated=False) as client:
        response = client.post(
            "/auth/register",
            json={"email": "existing@example.com", "password": "password123"},
        )

    assert response.status_code == 409


def test_session_job_is_scoped_to_owner(monkeypatch, fake_session_job_store):
    import app.routes.interview as interview_routes

    monkeypatch.setattr(interview_routes, "extract_resume_text", lambda _: RESUME_TEXT)

    with _client_without_startup_db(monkeypatch, user_id=1) as client:
        create_response = client.post(
            "/sessions/jobs",
            data={
                "job_description": ENGLISH_JD,
                "role_type": "AI Engineer",
                "output_language": "English",
                "demo_mode": "true",
            },
            files={"resume_pdf": ("resume.pdf", b"%PDF demo", "application/pdf")},
        )
        job_id = create_response.json()["job_id"]
        owner_response = client.get(f"/sessions/jobs/{job_id}")

    with _client_without_startup_db(monkeypatch, user_id=2) as client:
        other_response = client.get(f"/sessions/jobs/{job_id}")

    assert owner_response.status_code == 200
    assert other_response.status_code == 404
