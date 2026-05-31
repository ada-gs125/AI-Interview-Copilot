from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

import psycopg


_MigrationFn = Callable[[psycopg.Connection], None]

logger = logging.getLogger(__name__)


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


def _003_add_pgvector_rag(conn: psycopg.Connection) -> None:
    available = conn.execute(
        "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
    ).fetchone()
    if available is None:
        logger.warning("pgvector extension not available in this PostgreSQL instance; RAG feature disabled")
        return

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS question_embeddings (
            id BIGSERIAL PRIMARY KEY,
            session_id BIGINT REFERENCES sessions(id) ON DELETE CASCADE,
            question_text TEXT NOT NULL,
            answer_text TEXT,
            embedding vector(1536),
            role_type TEXT NOT NULL,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    # B-tree index for fast per-user role-type filtering before the vector scan.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_question_embeddings_user_role "
        "ON question_embeddings (user_id, role_type)"
    )
    # HNSW index for sub-linear approximate nearest-neighbour search (pgvector >= 0.5.0).
    # Falls back gracefully on older installations via savepoint.
    conn.execute("SAVEPOINT hnsw_index")
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_question_embeddings_hnsw "
            "ON question_embeddings USING hnsw (embedding vector_cosine_ops)"
        )
        conn.execute("RELEASE SAVEPOINT hnsw_index")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT hnsw_index")
        conn.execute("RELEASE SAVEPOINT hnsw_index")
        logger.warning("HNSW index not available (requires pgvector >= 0.5.0); vector queries will use sequential scan")


_MIGRATIONS: list[tuple[str, _MigrationFn]] = [
    ("001_initial_schema", _001_initial_schema),
    ("002_add_user_auth", _002_add_user_auth),
    ("003_add_pgvector_rag", _003_add_pgvector_rag),
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
