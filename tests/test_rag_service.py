"""Tests for RAGQuestionBank: retrieval, storage, availability guard, and error handling."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import AnswerResult, InterviewQuestion
from app.services.rag_service import RAGQuestionBank
from tests.factories import sample_answers, sample_questions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(n: int = 1) -> MagicMock:
    """Return a mock OpenAI client whose embeddings.create returns n unit vectors."""
    client = MagicMock()
    client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1] * 1536, index=i) for i in range(n)
    ]
    return client


def _questions(n: int = 2) -> list[InterviewQuestion]:
    return [
        InterviewQuestion(question=f"Q{i}?", why_it_matters=f"Why {i}.", difficulty="medium")
        for i in range(n)
    ]


def _answers(questions: list[InterviewQuestion]) -> list[AnswerResult]:
    return [
        AnswerResult(
            category="Technical",
            question=q.question,
            concise_answer=f"Answer {q.question}",
            resume_evidence_used=["evidence"],
            honesty_guardrail="honest",
        )
        for q in questions
    ]


@contextmanager
def _db_ok(rows=None):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows or []
    yield conn


@contextmanager
def _db_failing(msg="db error"):
    conn = MagicMock()
    conn.execute.side_effect = Exception(msg)
    conn.executemany.side_effect = Exception(msg)
    yield conn


@pytest.fixture(autouse=True)
def reset_rag_cache():
    """Reset class-level table availability cache before and after each test."""
    RAGQuestionBank._table_available = None
    yield
    RAGQuestionBank._table_available = None


# ---------------------------------------------------------------------------
# _check_table_available
# ---------------------------------------------------------------------------

class TestCheckTableAvailable:
    def test_returns_true_and_caches_when_table_exists(self):
        rag = RAGQuestionBank(_mock_client())
        with patch("app.services.rag_service.get_connection", lambda: _db_ok()):
            assert rag._check_table_available() is True
        assert RAGQuestionBank._table_available is True

    def test_returns_false_and_caches_when_table_missing(self):
        rag = RAGQuestionBank(_mock_client())
        with patch("app.services.rag_service.get_connection", lambda: _db_failing("relation does not exist")):
            assert rag._check_table_available() is False
        assert RAGQuestionBank._table_available is False

    def test_uses_cached_result_without_db_call(self):
        RAGQuestionBank._table_available = True
        rag = RAGQuestionBank(_mock_client())
        with patch("app.services.rag_service.get_connection") as mock_gc:
            assert rag._check_table_available() is True
            mock_gc.assert_not_called()


# ---------------------------------------------------------------------------
# retrieve_similar
# ---------------------------------------------------------------------------

class TestRetrieveSimilar:
    def test_skips_embed_and_returns_empty_when_table_unavailable(self):
        RAGQuestionBank._table_available = False
        client = _mock_client()
        result = RAGQuestionBank(client).retrieve_similar("AI Engineer", ["Python"], user_id=1)
        assert result == []
        client.embeddings.create.assert_not_called()

    def test_returns_empty_on_embed_error(self):
        RAGQuestionBank._table_available = True
        client = _mock_client()
        client.embeddings.create.side_effect = Exception("network error")
        assert RAGQuestionBank(client).retrieve_similar("AI Engineer", ["Python"], user_id=1) == []

    def test_returns_empty_on_db_error(self):
        RAGQuestionBank._table_available = True
        rag = RAGQuestionBank(_mock_client())
        with patch("app.services.rag_service.get_connection", lambda: _db_failing()):
            assert rag.retrieve_similar("AI Engineer", ["Python"], user_id=1) == []

    def test_user_id_appears_in_query_params(self):
        RAGQuestionBank._table_available = True
        rag = RAGQuestionBank(_mock_client())
        captured: list[tuple] = []

        @contextmanager
        def spy_conn():
            conn = MagicMock()
            def capture(sql, params):
                captured.append(params)
                return MagicMock(fetchall=lambda: [])
            conn.execute.side_effect = capture
            yield conn

        with patch("app.services.rag_service.get_connection", spy_conn):
            rag.retrieve_similar("AI Engineer", ["Python"], user_id=42)

        assert captured, "execute was never called"
        flat = [v for row in captured for v in row]
        assert 42 in flat, "user_id=42 should appear in query params"

    def test_maps_rows_to_dicts(self):
        RAGQuestionBank._table_available = True
        rag = RAGQuestionBank(_mock_client())
        fake_rows = [
            {"question_text": "Tell me about X?", "answer_text": "I did X.", "similarity": 0.92}
        ]
        with patch("app.services.rag_service.get_connection", lambda: _db_ok(fake_rows)):
            results = rag.retrieve_similar("AI Engineer", ["Python"], user_id=1)
        assert len(results) == 1
        assert results[0]["question"] == "Tell me about X?"
        assert results[0]["answer"] == "I did X."
        assert results[0]["score"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class TestStore:
    def test_returns_zero_for_empty_questions(self):
        rag = RAGQuestionBank(_mock_client())
        assert rag.store(1, [], [], "AI Engineer", user_id=1) == 0

    def test_skips_embed_and_returns_zero_when_table_unavailable(self):
        RAGQuestionBank._table_available = False
        client = _mock_client()
        qs = _questions(2)
        assert RAGQuestionBank(client).store(1, qs, _answers(qs), "AI Engineer", user_id=1) == 0
        client.embeddings.create.assert_not_called()

    def test_returns_zero_on_embed_error(self):
        RAGQuestionBank._table_available = True
        client = _mock_client()
        client.embeddings.create.side_effect = Exception("embed error")
        qs = _questions(2)
        assert RAGQuestionBank(client).store(1, qs, _answers(qs), "AI Engineer", user_id=1) == 0

    def test_returns_zero_on_db_error(self):
        RAGQuestionBank._table_available = True
        n = 2
        qs = _questions(n)
        rag = RAGQuestionBank(_mock_client(n))
        with patch("app.services.rag_service.get_connection", lambda: _db_failing()):
            assert rag.store(1, qs, _answers(qs), "AI Engineer", user_id=1) == 0

    def test_returns_count_and_calls_executemany_once(self):
        RAGQuestionBank._table_available = True
        n = 3
        qs = _questions(n)
        rag = RAGQuestionBank(_mock_client(n))
        batches: list[list] = []

        @contextmanager
        def capture_conn():
            conn = MagicMock()
            conn.executemany.side_effect = lambda _sql, rows: batches.append(list(rows))
            yield conn

        with patch("app.services.rag_service.get_connection", capture_conn):
            count = rag.store(1, qs, _answers(qs), "AI Engineer", user_id=7)

        assert count == n
        assert len(batches) == 1, "expected exactly one executemany call"
        assert len(batches[0]) == n, "all rows should be in a single batch"
        # user_id is the last element in every row tuple
        for row in batches[0]:
            assert row[-1] == 7

    def test_uses_answer_map_to_populate_answer_text(self):
        RAGQuestionBank._table_available = True
        qs = sample_questions().technical_questions  # 1 question
        ans = sample_answers().answers
        rag = RAGQuestionBank(_mock_client(len(qs)))
        stored_rows: list[list] = []

        @contextmanager
        def capture_conn():
            conn = MagicMock()
            conn.executemany.side_effect = lambda _sql, rows: stored_rows.append(list(rows))
            yield conn

        with patch("app.services.rag_service.get_connection", capture_conn):
            rag.store(1, qs, ans, "AI Engineer", user_id=1)

        assert stored_rows
        # Row layout: (session_id, question_text, answer_text, embedding_json, role_type, user_id)
        _session_id, _q, answer_text, _emb, _role, _uid = stored_rows[0][0]
        assert "Pydantic" in answer_text  # from sample_answers concise_answer

    def test_raises_assertion_on_embedding_count_mismatch(self):
        RAGQuestionBank._table_available = True
        qs = _questions(3)
        # Only 1 embedding returned for 3 questions
        rag = RAGQuestionBank(_mock_client(1))
        with pytest.raises(AssertionError, match="Embedding count mismatch"):
            rag.store(1, qs, _answers(qs), "AI Engineer", user_id=1)
