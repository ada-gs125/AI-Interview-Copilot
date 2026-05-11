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
    JobError,
    JobStep,
    JobStatus,
    OutputLanguage,
    QuestionSet,
    ResumeMatch,
    SessionJobResponse,
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                current_step TEXT,
                progress_percent INTEGER NOT NULL DEFAULT 0,
                role_type TEXT NOT NULL,
                output_language TEXT NOT NULL DEFAULT 'Match job description language',
                demo_mode BOOLEAN NOT NULL DEFAULT FALSE,
                session_id BIGINT REFERENCES sessions(id) ON DELETE SET NULL,
                error JSONB,
                steps JSONB NOT NULL DEFAULT '[]'::jsonb,
                usage JSONB NOT NULL DEFAULT '{}'::jsonb,
                result JSONB
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_jobs_created_at
            ON session_jobs (created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_jobs_status
            ON session_jobs (status)
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


def create_session_job(
    *,
    job_id: str,
    role_type: str,
    output_language: OutputLanguage,
    demo_mode: bool,
) -> str:
    now = datetime.now(timezone.utc)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO session_jobs (
                id, status, created_at, updated_at, role_type, output_language,
                demo_mode, steps, usage
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                job_id,
                "queued",
                now,
                now,
                role_type,
                output_language,
                demo_mode,
                Jsonb([]),
                Jsonb({}),
            ),
        )
    return job_id


def update_session_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    current_step: Optional[str] = None,
    progress_percent: Optional[int] = None,
    session_id: Optional[int] = None,
    error: Optional[dict[str, Any]] = None,
    steps: Optional[list[dict[str, Any]]] = None,
    usage: Optional[dict[str, Any]] = None,
    result: Optional[SessionResponse] = None,
    completed: bool = False,
) -> None:
    assignments = ["updated_at = %s"]
    values: list[Any] = [datetime.now(timezone.utc)]

    if status is not None:
        assignments.append("status = %s")
        values.append(status)
    if current_step is not None:
        assignments.append("current_step = %s")
        values.append(current_step)
    if progress_percent is not None:
        assignments.append("progress_percent = %s")
        values.append(progress_percent)
    if session_id is not None:
        assignments.append("session_id = %s")
        values.append(session_id)
    if error is not None:
        assignments.append("error = %s")
        values.append(Jsonb(error))
    if steps is not None:
        assignments.append("steps = %s")
        values.append(Jsonb(steps))
    if usage is not None:
        assignments.append("usage = %s")
        values.append(Jsonb(usage))
    if result is not None:
        assignments.append("result = %s")
        values.append(Jsonb(result.model_dump(mode="json")))
    if completed:
        assignments.append("completed_at = %s")
        values.append(datetime.now(timezone.utc))

    values.append(job_id)
    with get_connection() as conn:
        conn.execute(
            f"""
            UPDATE session_jobs
            SET {", ".join(assignments)}
            WHERE id = %s
            """,
            values,
        )


def _row_to_job(row: DictRow) -> SessionJobResponse:
    error = JobError.model_validate(row["error"]) if row["error"] else None
    result = SessionResponse.model_validate(row["result"]) if row["result"] else None
    return SessionJobResponse(
        id=row["id"],
        status=row["status"],
        created_at=_created_at_to_iso(row["created_at"]),
        updated_at=_created_at_to_iso(row["updated_at"]),
        completed_at=_created_at_to_iso(row["completed_at"]) if row["completed_at"] else None,
        current_step=row["current_step"],
        progress_percent=row["progress_percent"],
        role_type=row["role_type"],
        output_language=row["output_language"] or DEFAULT_OUTPUT_LANGUAGE,
        demo_mode=bool(row["demo_mode"]),
        session_id=row["session_id"],
        error=error,
        steps=[JobStep.model_validate(step) for step in row["steps"]],
        usage=row["usage"] or {},
        result=result,
    )


def get_session_job(job_id: str) -> Optional[SessionJobResponse]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM session_jobs WHERE id = %s", (job_id,)).fetchone()
    return _row_to_job(row) if row else None
