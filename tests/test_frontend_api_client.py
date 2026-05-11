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

    def fake_post(url, *, files, data, timeout):
        calls.append((url, files, data, timeout))
        return DummyResponse({"job_id": "job-123", "status_url": "/sessions/jobs/job-123"})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    payload = api_client.create_session_job(
        "https://api.example.test",
        DummyResumeFile(),
        "We need an AI backend engineer with Python, APIs, persistence, and LLM workflow experience.",
        "AI Engineer",
        "English",
        True,
    )

    assert payload["job_id"] == "job-123"
    url, files, data, timeout = calls[0]
    assert url == "https://api.example.test/sessions/jobs"
    assert files["resume_pdf"][0] == "resume.pdf"
    assert data["demo_mode"] is True
    assert timeout == 30


def test_create_session_from_upload_posts_to_sync_endpoint(monkeypatch):
    calls = []

    def fake_post(url, *, files, data, timeout):
        calls.append((url, files, data, timeout))
        return DummyResponse({"id": 0, "demo_mode": True})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    payload = api_client.create_session_from_upload(
        "https://api.example.test",
        DummyResumeFile(),
        "We need an AI backend engineer with Python, APIs, persistence, and LLM workflow experience.",
        "AI Engineer",
        "English",
        True,
    )

    assert payload["demo_mode"] is True
    assert calls[0][0] == "https://api.example.test/sessions/from-upload"
    assert calls[0][3] == 240


def test_get_session_job_uses_status_url(monkeypatch):
    def fake_get(url, *, timeout):
        assert url == "https://api.example.test/sessions/jobs/job-123"
        assert timeout == 30
        return DummyResponse({"id": "job-123", "status": "succeeded"})

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    assert api_client.get_session_job("https://api.example.test", "/sessions/jobs/job-123")["status"] == "succeeded"


def test_fallback_to_sync_only_for_missing_job_api():
    missing = requests.HTTPError("missing")
    missing.response = DummyResponse(status_code=404)
    method_missing = requests.HTTPError("method missing")
    method_missing.response = DummyResponse(status_code=405)
    server_error = requests.HTTPError("server")
    server_error.response = DummyResponse(status_code=500)

    assert api_client.should_fallback_to_sync(missing) is True
    assert api_client.should_fallback_to_sync(method_missing) is True
    assert api_client.should_fallback_to_sync(server_error) is False


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
