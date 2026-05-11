import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest

from app.config import get_settings
from app.database import db
from tests.factories import (
    ENGLISH_JD,
    RESUME_TEXT,
    sample_answers,
    sample_jd_analysis,
    sample_questions,
    sample_resume_match,
    sample_session,
)


def _database_url_with_search_path(database_url: str, schema: str) -> str:
    parts = urlsplit(database_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["options"] = f"-csearch_path={schema}"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


@pytest.fixture
def postgres_database(monkeypatch):
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests.")

    schema = f"test_{uuid4().hex}"
    test_database_url = _database_url_with_search_path(database_url, schema)

    monkeypatch.setenv("DATABASE_URL", test_database_url)
    get_settings.cache_clear()
    try:
        with psycopg.connect(database_url, autocommit=True) as admin_conn:
            admin_conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        db.init_db()
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL is not available: {exc}")

    yield

    with psycopg.connect(database_url, autocommit=True) as admin_conn:
        admin_conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
    get_settings.cache_clear()


def test_create_get_and_list_session_round_trip(postgres_database):
    user = db.create_user(email=f"user-{uuid4().hex}@example.com", password="password123")
    session_id = db.create_session(
        user_id=user.id,
        role_type="AI Engineer",
        output_language="English",
        demo_mode=False,
        job_description=ENGLISH_JD,
        resume_text=RESUME_TEXT,
        jd_analysis=sample_jd_analysis(),
        resume_match=sample_resume_match(),
        questions=sample_questions(),
        answers=sample_answers(),
    )

    loaded = db.get_session(session_id, user_id=user.id)
    summaries = db.list_sessions(user_id=user.id)

    assert loaded is not None
    assert loaded.id == session_id
    assert loaded.user_id == user.id
    assert loaded.created_at.endswith("+00:00")
    assert loaded.jd_analysis.required_technical_skills[0].name == "Python"
    assert loaded.resume_match.overall_fit_score == 82
    assert loaded.questions.technical_questions[0].difficulty == "hard"
    assert loaded.answers.answers[0].resume_evidence_used == ["Interview Copilot project"]
    assert summaries[0].id == session_id
    assert summaries[0].user_id == user.id
    assert summaries[0].missing_skill_count == 1


def test_get_session_returns_none_for_missing_id(postgres_database):
    user = db.create_user(email=f"user-{uuid4().hex}@example.com", password="password123")
    assert db.get_session(999_999, user_id=user.id) is None


def test_create_update_and_get_session_job_round_trip(postgres_database):
    user = db.create_user(email=f"user-{uuid4().hex}@example.com", password="password123")
    job_id = f"job-{uuid4().hex}"
    db.create_session_job(
        job_id=job_id,
        user_id=user.id,
        role_type="AI Engineer",
        output_language="English",
        demo_mode=True,
    )

    db.update_session_job(
        job_id,
        status="succeeded",
        current_step="completed",
        progress_percent=100,
        steps=[
            {
                "name": "parse_resume",
                "status": "succeeded",
                "started_at": "2026-05-11T00:00:00+00:00",
                "completed_at": "2026-05-11T00:00:01+00:00",
                "latency_ms": 10,
                "usage": {},
            }
        ],
        usage={"call_count": 0, "calls": []},
        result=sample_session(session_id=0),
        completed=True,
    )

    loaded = db.get_session_job(job_id, user_id=user.id)

    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.user_id == user.id
    assert loaded.progress_percent == 100
    assert loaded.steps[0].name == "parse_resume"
    assert loaded.result is not None
    assert loaded.result.demo_mode is False


def test_user_auth_and_session_isolation(postgres_database):
    user_a = db.create_user(email=f"user-{uuid4().hex}@example.com", password="password123")
    user_b = db.create_user(email=f"user-{uuid4().hex}@example.com", password="password123")

    assert db.authenticate_user(email=user_a.email, password="password123") is not None
    assert db.authenticate_user(email=user_a.email, password="wrong-password") is None

    session_id = db.create_session(
        user_id=user_a.id,
        role_type="AI Engineer",
        output_language="English",
        demo_mode=False,
        job_description=ENGLISH_JD,
        resume_text=RESUME_TEXT,
        jd_analysis=sample_jd_analysis(),
        resume_match=sample_resume_match(),
        questions=sample_questions(),
        answers=sample_answers(),
    )

    assert db.get_session(session_id, user_id=user_a.id) is not None
    assert db.get_session(session_id, user_id=user_b.id) is None
    assert db.list_sessions(user_id=user_b.id) == []
    assert db.delete_session(session_id, user_id=user_b.id) is False
    assert db.delete_session(session_id, user_id=user_a.id) is True
