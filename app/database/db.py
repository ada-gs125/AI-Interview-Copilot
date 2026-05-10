from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import psycopg
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

from app.config import get_settings
from app.schemas import (
    AnswerSet,
    JDAnalysis,
    OutputLanguage,
    QuestionSet,
    ResumeMatch,
    SessionResponse,
    SessionSummary,
)


DEFAULT_OUTPUT_LANGUAGE: OutputLanguage = "Match job description language"


@contextmanager
def get_connection() -> Iterator[psycopg.Connection[DictRow]]:
    conn = psycopg.connect(get_settings().database_url, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL,
                role_type TEXT NOT NULL,
                output_language TEXT NOT NULL DEFAULT 'Match job description language',
                demo_mode BOOLEAN NOT NULL DEFAULT FALSE,
                job_description TEXT NOT NULL,
                resume_text TEXT NOT NULL,
                jd_analysis JSONB NOT NULL,
                resume_match JSONB NOT NULL,
                questions JSONB NOT NULL,
                answers JSONB NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_created_at
            ON sessions (created_at DESC)
            """
        )


def create_session(
    *,
    role_type: str,
    output_language: OutputLanguage,
    demo_mode: bool,
    job_description: str,
    resume_text: str,
    jd_analysis: JDAnalysis,
    resume_match: ResumeMatch,
    questions: QuestionSet,
    answers: AnswerSet,
) -> int:
    created_at = datetime.now(timezone.utc)
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO sessions (
                created_at, role_type, output_language, demo_mode, job_description, resume_text,
                jd_analysis, resume_match, questions, answers
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                created_at,
                role_type,
                output_language,
                demo_mode,
                job_description,
                resume_text,
                Jsonb(jd_analysis.model_dump(mode="json")),
                Jsonb(resume_match.model_dump(mode="json")),
                Jsonb(questions.model_dump(mode="json")),
                Jsonb(answers.model_dump(mode="json")),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL did not return a session id.")
        return int(row["id"])


def _created_at_to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _row_to_session(row: DictRow) -> SessionResponse:
    return SessionResponse(
        id=row["id"],
        created_at=_created_at_to_iso(row["created_at"]),
        role_type=row["role_type"],
        output_language=row["output_language"] or DEFAULT_OUTPUT_LANGUAGE,
        demo_mode=bool(row["demo_mode"]),
        job_description=row["job_description"],
        resume_text=row["resume_text"],
        jd_analysis=JDAnalysis.model_validate(row["jd_analysis"]),
        resume_match=ResumeMatch.model_validate(row["resume_match"]),
        questions=QuestionSet.model_validate(row["questions"]),
        answers=AnswerSet.model_validate(row["answers"]),
    )


def get_session(session_id: int) -> Optional[SessionResponse]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = %s", (session_id,)).fetchone()
    return _row_to_session(row) if row else None


def list_sessions(limit: int = 25) -> list[SessionSummary]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM sessions
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()

    summaries: list[SessionSummary] = []
    for row in rows:
        jd_analysis = JDAnalysis.model_validate(row["jd_analysis"])
        resume_match = ResumeMatch.model_validate(row["resume_match"])
        summaries.append(
            SessionSummary(
                id=row["id"],
                created_at=_created_at_to_iso(row["created_at"]),
                role_type=row["role_type"],
                output_language=row["output_language"] or DEFAULT_OUTPUT_LANGUAGE,
                demo_mode=bool(row["demo_mode"]),
                overall_fit_score=resume_match.overall_fit_score,
                role_summary=jd_analysis.role_summary,
                missing_skill_count=len(resume_match.missing_skills),
            )
        )
    return summaries
