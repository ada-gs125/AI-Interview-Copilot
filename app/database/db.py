from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import psycopg
from psycopg import errors
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
    UserResponse,
)
from app.services.auth_service import hash_password, normalize_email, verify_password


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
    from app.database.migrations import run_migrations
    with get_connection() as conn:
        run_migrations(conn)


def _row_to_user(row: DictRow) -> UserResponse:
    return UserResponse(
        id=row["id"],
        email=row["email"],
        created_at=_created_at_to_iso(row["created_at"]),
    )


def create_user(*, email: str, password: str) -> UserResponse:
    normalized_email = normalize_email(email)
    created_at = datetime.now(timezone.utc)
    try:
        with get_connection() as conn:
            row = conn.execute(
                """
                INSERT INTO users (email, password_hash, created_at)
                VALUES (%s, %s, %s)
                RETURNING id, email, created_at
                """,
                (normalized_email, hash_password(password), created_at),
            ).fetchone()
    except errors.UniqueViolation as exc:
        raise ValueError("A user with this email already exists.") from exc
    if row is None:
        raise RuntimeError("PostgreSQL did not return a user.")
    return _row_to_user(row)


def get_user(user_id: int) -> Optional[UserResponse]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, email, created_at FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()
    return _row_to_user(row) if row else None


def get_user_credentials_by_email(email: str) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, email, password_hash, created_at
            FROM users
            WHERE lower(email) = lower(%s)
            """,
            (normalize_email(email),),
        ).fetchone()
    return dict(row) if row else None


def authenticate_user(*, email: str, password: str) -> Optional[UserResponse]:
    row = get_user_credentials_by_email(email)
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return UserResponse(
        id=row["id"],
        email=row["email"],
        created_at=_created_at_to_iso(row["created_at"]),
    )


def create_session(
    *,
    user_id: int,
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
                jd_analysis, resume_match, questions, answers, user_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                user_id,
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
        user_id=row["user_id"],
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


def get_session(session_id: int, *, user_id: int) -> Optional[SessionResponse]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = %s AND user_id = %s",
            (session_id, user_id),
        ).fetchone()
    return _row_to_session(row) if row else None


def list_sessions(*, user_id: int, limit: int = 25) -> list[SessionSummary]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, created_at, role_type, output_language, demo_mode, jd_analysis, resume_match
            FROM sessions
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()

    summaries: list[SessionSummary] = []
    for row in rows:
        jd_analysis = JDAnalysis.model_validate(row["jd_analysis"])
        resume_match = ResumeMatch.model_validate(row["resume_match"])
        summaries.append(
            SessionSummary(
                id=row["id"],
                created_at=_created_at_to_iso(row["created_at"]),
                user_id=row["user_id"],
                role_type=row["role_type"],
                output_language=row["output_language"] or DEFAULT_OUTPUT_LANGUAGE,
                demo_mode=bool(row["demo_mode"]),
                overall_fit_score=resume_match.overall_fit_score,
                role_summary=jd_analysis.role_summary,
                missing_skill_count=len(resume_match.missing_skills),
            )
        )
    return summaries


def delete_session(session_id: int, *, user_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            """
            DELETE FROM sessions
            WHERE id = %s AND user_id = %s
            RETURNING id
            """,
            (session_id, user_id),
        ).fetchone()
    return row is not None


def delete_expired_sessions(*, user_id: int, retention_days: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    with get_connection() as conn:
        rows = conn.execute(
            """
            DELETE FROM sessions
            WHERE user_id = %s AND created_at < %s
            RETURNING id
            """,
            (user_id, cutoff),
        ).fetchall()
    return len(rows)


def create_session_job(
    *,
    job_id: str,
    user_id: int,
    role_type: str,
    output_language: OutputLanguage,
    demo_mode: bool,
) -> str:
    now = datetime.now(timezone.utc)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO session_jobs (
                id, user_id, status, created_at, updated_at, role_type, output_language,
                demo_mode, steps, usage
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                job_id,
                user_id,
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
        user_id=row["user_id"],
        session_id=row["session_id"],
        error=error,
        steps=[JobStep.model_validate(step) for step in row["steps"]],
        usage=row["usage"] or {},
        result=result,
    )


def get_session_job(job_id: str, *, user_id: int) -> Optional[SessionJobResponse]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM session_jobs WHERE id = %s AND user_id = %s",
            (job_id, user_id),
        ).fetchone()
    return _row_to_job(row) if row else None
