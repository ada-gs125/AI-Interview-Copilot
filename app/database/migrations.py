from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import psycopg


_MigrationFn = Callable[[psycopg.Connection], None]


def _001_initial_schema(conn: psycopg.Connection) -> None:
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
        "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions (created_at DESC)"
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
        "CREATE INDEX IF NOT EXISTS idx_session_jobs_created_at ON session_jobs (created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_jobs_status ON session_jobs (status)"
    )


def _002_add_user_auth(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_lower ON users (lower(email))"
    )
    conn.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_user_created_at ON sessions (user_id, created_at DESC)"
    )
    conn.execute(
        "ALTER TABLE session_jobs ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_jobs_user_created_at ON session_jobs (user_id, created_at DESC)"
    )


_MIGRATIONS: list[tuple[str, _MigrationFn]] = [
    ("001_initial_schema", _001_initial_schema),
    ("002_add_user_auth", _002_add_user_auth),
]


def run_migrations(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    applied = {
        row["version"]
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for version, fn in _MIGRATIONS:
        if version not in applied:
            fn(conn)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (%s, %s)",
                (version, datetime.now(timezone.utc)),
            )
