from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from app.config import get_settings
from app.schemas import AnswerSet, JDAnalysis, QuestionSet, ResumeMatch, SessionResponse, SessionSummary


def _db_path() -> Path:
    path = get_settings().database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                role_type TEXT NOT NULL,
                job_description TEXT NOT NULL,
                resume_text TEXT NOT NULL,
                jd_analysis TEXT NOT NULL,
                resume_match TEXT NOT NULL,
                questions TEXT NOT NULL,
                answers TEXT NOT NULL
            )
            """
        )


def create_session(
    *,
    role_type: str,
    job_description: str,
    resume_text: str,
    jd_analysis: JDAnalysis,
    resume_match: ResumeMatch,
    questions: QuestionSet,
    answers: AnswerSet,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO sessions (
                created_at, role_type, job_description, resume_text,
                jd_analysis, resume_match, questions, answers
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                role_type,
                job_description,
                resume_text,
                jd_analysis.model_dump_json(),
                resume_match.model_dump_json(),
                questions.model_dump_json(),
                answers.model_dump_json(),
            ),
        )
        return int(cursor.lastrowid)


def _row_to_session(row: sqlite3.Row) -> SessionResponse:
    return SessionResponse(
        id=row["id"],
        created_at=row["created_at"],
        role_type=row["role_type"],
        job_description=row["job_description"],
        resume_text=row["resume_text"],
        jd_analysis=JDAnalysis.model_validate(json.loads(row["jd_analysis"])),
        resume_match=ResumeMatch.model_validate(json.loads(row["resume_match"])),
        questions=QuestionSet.model_validate(json.loads(row["questions"])),
        answers=AnswerSet.model_validate(json.loads(row["answers"])),
    )


def get_session(session_id: int) -> Optional[SessionResponse]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_session(row) if row else None


def list_sessions(limit: int = 25) -> list[SessionSummary]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY datetime(created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()

    summaries: list[SessionSummary] = []
    for row in rows:
        jd_analysis = JDAnalysis.model_validate(json.loads(row["jd_analysis"]))
        resume_match = ResumeMatch.model_validate(json.loads(row["resume_match"]))
        summaries.append(
            SessionSummary(
                id=row["id"],
                created_at=row["created_at"],
                role_type=row["role_type"],
                overall_fit_score=resume_match.overall_fit_score,
                role_summary=jd_analysis.role_summary,
                missing_skill_count=len(resume_match.missing_skills),
            )
        )
    return summaries
