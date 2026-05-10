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
    session_id = db.create_session(
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

    loaded = db.get_session(session_id)
    summaries = db.list_sessions()

    assert loaded is not None
    assert loaded.id == session_id
    assert loaded.created_at.endswith("+00:00")
    assert loaded.jd_analysis.required_technical_skills[0].name == "Python"
    assert loaded.resume_match.overall_fit_score == 82
    assert loaded.questions.technical_questions[0].difficulty == "hard"
    assert loaded.answers.answers[0].resume_evidence_used == ["Interview Copilot project"]
    assert summaries[0].id == session_id
    assert summaries[0].missing_skill_count == 1


def test_get_session_returns_none_for_missing_id(postgres_database):
    assert db.get_session(999_999) is None
