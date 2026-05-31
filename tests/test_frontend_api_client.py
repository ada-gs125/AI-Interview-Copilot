from __future__ import annotations

from dataclasses import dataclass

import requests

from app.frontend import api_client


@dataclass
class DummyResumeFile:
    name: str = "resume.pdf"

    def getvalue(self) -> bytes:
        return b"%PDF demo"


class DummyResponse:
    def __init__(self, payload=None, *, status_code: int = 200, text: str = "") -> None:
        self.payload = payload or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error


def test_create_session_job_posts_to_job_endpoint(monkeypatch):
    calls = []

    def fake_post(url, *, files, data, headers, timeout):
        calls.append((url, files, data, headers, timeout))
        return DummyResponse({"job_id": "job-123", "status_url": "/sessions/jobs/job-123"})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    payload = api_client.create_session_job(
        "https://api.example.test",
        DummyResumeFile(),
        "We need an AI backend engineer with Python, APIs, persistence, and LLM workflow experience.",
        "AI Engineer",
        "English",
        True,
        "token-123",
    )

    assert payload["job_id"] == "job-123"
    url, files, data, headers, timeout = calls[0]
    assert url == "https://api.example.test/sessions/jobs"
    assert files["resume_pdf"][0] == "resume.pdf"
    assert data["demo_mode"] is True
    assert headers == {"Authorization": "Bearer token-123"}
    assert timeout == 30


def test_get_session_job_uses_status_url(monkeypatch):
    def fake_get(url, *, headers, timeout):
        assert url == "https://api.example.test/sessions/jobs/job-123"
        assert headers == {"Authorization": "Bearer token-123"}
        assert timeout == 30
        return DummyResponse({"id": "job-123", "status": "succeeded"})

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    assert (
        api_client.get_session_job("https://api.example.test", "/sessions/jobs/job-123", "token-123")["status"]
        == "succeeded"
    )


def test_auth_endpoints_send_json(monkeypatch):
    calls = []

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return DummyResponse({"access_token": "token-123", "user": {"email": json["email"]}})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    register_payload = api_client.register_user("https://api.example.test", "user@example.com", "password123")
    login_payload = api_client.login_user("https://api.example.test", "user@example.com", "password123")

    assert register_payload["access_token"] == "token-123"
    assert login_payload["access_token"] == "token-123"
    assert calls[0][0] == "https://api.example.test/auth/register"
    assert calls[1][0] == "https://api.example.test/auth/login"


def test_api_delete_sends_auth_header(monkeypatch):
    calls = []

    def fake_delete(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return DummyResponse(status_code=204)

    monkeypatch.setattr(api_client.requests, "delete", fake_delete)

    api_client.api_delete("https://api.example.test", "/sessions/42", "token-123")

    assert calls == [("https://api.example.test/sessions/42", {"Authorization": "Bearer token-123"}, 30)]


def test_api_get_sends_auth_header(monkeypatch):
    calls = []

    def fake_get(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return DummyResponse({"id": 42, "role_type": "AI Engineer"})

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    payload = api_client.api_get("https://api.example.test", "/sessions/42", "token-123")

    assert payload == {"id": 42, "role_type": "AI Engineer"}
    assert calls == [("https://api.example.test/sessions/42", {"Authorization": "Bearer token-123"}, 30)]


def test_friendly_api_error_extracts_structured_detail():
    exc = requests.HTTPError("bad request")
    exc.response = DummyResponse(
        {
            "detail": {
                "message": "Could not extract enough text from the PDF resume.",
                "action": "Upload a text-based PDF resume.",
                "code": "pdf_parse_error",
            }
        },
        status_code=400,
    )

    message, action = api_client.friendly_api_error(exc)

    assert message == "Could not extract enough text from the PDF resume."
    assert action == "Upload a text-based PDF resume."
