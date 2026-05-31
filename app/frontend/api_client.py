from __future__ import annotations

import json
from typing import Any, Iterator

import requests


def auth_headers(access_token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"} if access_token else {}


def api_get(api_base_url: str, path: str, access_token: str | None = None) -> Any:
    response = requests.get(f"{api_base_url}{path}", headers=auth_headers(access_token), timeout=30)
    response.raise_for_status()
    return response.json()


def api_delete(api_base_url: str, path: str, access_token: str | None = None) -> None:
    response = requests.delete(f"{api_base_url}{path}", headers=auth_headers(access_token), timeout=30)
    response.raise_for_status()


def register_user(api_base_url: str, email: str, password: str) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url}/auth/register",
        json={"email": email, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def login_user(api_base_url: str, email: str, password: str) -> dict[str, Any]:
    response = requests.post(
        f"{api_base_url}/auth/login",
        json={"email": email, "password": password},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _upload_payload(
    resume_file,
    job_description: str,
    role_type: str,
    output_language: str,
    demo_mode: bool,
) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, Any]]:
    files = {"resume_pdf": (resume_file.name, resume_file.getvalue(), "application/pdf")}
    data = {
        "job_description": job_description,
        "role_type": role_type,
        "output_language": output_language,
        "demo_mode": demo_mode,
    }
    return files, data


def create_session_job(
    api_base_url: str,
    resume_file,
    job_description: str,
    role_type: str,
    output_language: str,
    demo_mode: bool,
    access_token: str | None = None,
) -> dict[str, Any]:
    files, data = _upload_payload(
        resume_file,
        job_description,
        role_type,
        output_language,
        demo_mode,
    )
    response = requests.post(
        f"{api_base_url}/sessions/jobs",
        files=files,
        data=data,
        headers=auth_headers(access_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()



def get_session_job(api_base_url: str, status_url: str, access_token: str | None = None) -> dict[str, Any]:
    return api_get(api_base_url, status_url, access_token)


def stream_answer(
    api_base_url: str,
    payload: dict[str, Any],
    access_token: str | None = None,
) -> Iterator[str]:
    with requests.post(
        f"{api_base_url}/generate-answer/stream",
        json=payload,
        headers=auth_headers(access_token),
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or line == b"data: [DONE]":
                break
            if line.startswith(b"data: "):
                yield json.loads(line[6:])



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
