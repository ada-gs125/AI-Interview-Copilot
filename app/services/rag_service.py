"""pgvector-backed semantic question bank for few-shot question generation."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.database.db import get_connection
from app.schemas import AnswerResult, InterviewQuestion

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"


class RAGQuestionBank:
    """Semantic question bank backed by pgvector.

    Stores question+answer embeddings after each session and retrieves
    semantically similar examples before question generation to use as
    few-shot calibration context, reducing LLM call cost for repeated
    role types over time.

    All data is scoped to a single user — questions from one user's sessions
    are never surfaced in another user's retrieval results.
    """

    # Class-level cache: None = unchecked, True/False = result of first check.
    _table_available: bool | None = None

    def __init__(self, client: OpenAI) -> None:
        self.client = client

    # ------------------------------------------------------------------
    # Availability guard — checked once, cached for the process lifetime
    # ------------------------------------------------------------------

    @classmethod
    def _check_table_available(cls) -> bool:
        """Return True if the question_embeddings table exists and is queryable."""
        if cls._table_available is not None:
            return cls._table_available
        try:
            with get_connection() as conn:
                conn.execute("SELECT 1 FROM question_embeddings LIMIT 0")
            cls._table_available = True
        except Exception:
            cls._table_available = False
        return cls._table_available

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=[t[:8000] for t in texts],
        )
        # Sort by index defensively; OpenAI guarantees order but it costs nothing.
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_similar(
        self,
        role_type: str,
        skills: list[str],
        *,
        user_id: int,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return up to k past questions most similar to the given role+skills.

        Results are scoped to user_id; no cross-user data is returned.
        Returns [] silently on any infrastructure failure.
        """
        if not self._check_table_available():
            return []

        query = f"{role_type}: {', '.join(skills[:10])}"
        try:
            [embedding] = self._embed_batch([query])
        except Exception as exc:
            logger.warning("rag_embed_failed query=%r error=%s", query[:80], exc)
            return []

        embedding_str = json.dumps(embedding)
        try:
            with get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT question_text, answer_text,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM question_embeddings
                    WHERE user_id = %s AND role_type = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_str, user_id, role_type, embedding_str, k),
                ).fetchall()
        except Exception as exc:
            logger.warning("rag_retrieve_failed role_type=%r user_id=%d error=%s", role_type, user_id, exc)
            return []

        return [
            {
                "question": row["question_text"],
                "answer": row["answer_text"],
                "score": float(row["similarity"]),
            }
            for row in rows
        ]

    def store(
        self,
        session_id: int,
        questions: list[InterviewQuestion],
        answers: list[AnswerResult],
        role_type: str,
        *,
        user_id: int,
    ) -> int:
        """Batch-embed and store all questions scoped to user_id.

        Returns count stored, 0 on any failure.
        """
        if not questions or not self._check_table_available():
            return 0

        answer_map = {ans.question: ans.concise_answer for ans in answers}
        embed_texts = [f"{q.question} {q.why_it_matters}" for q in questions]

        try:
            embeddings = self._embed_batch(embed_texts)
        except Exception as exc:
            logger.warning("rag_store_embed_failed role_type=%r error=%s", role_type, exc)
            return 0

        assert len(embeddings) == len(questions), (
            f"Embedding count mismatch: got {len(embeddings)}, expected {len(questions)}"
        )

        try:
            with get_connection() as conn:
                conn.executemany(
                    """
                    INSERT INTO question_embeddings
                        (session_id, question_text, answer_text, embedding, role_type, user_id)
                    VALUES (%s, %s, %s, %s::vector, %s, %s)
                    """,
                    [
                        (session_id, q.question, answer_map.get(q.question), json.dumps(emb), role_type, user_id)
                        for q, emb in zip(questions, embeddings)
                    ],
                )
        except Exception as exc:
            logger.warning("rag_store_insert_failed session_id=%d error=%s", session_id, exc)
            return 0

        return len(questions)
