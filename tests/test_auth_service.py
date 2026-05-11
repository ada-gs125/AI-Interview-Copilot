from __future__ import annotations

import pytest

from app.services.auth_service import create_access_token, decode_access_token, hash_password, normalize_email, verify_password


def test_password_hash_round_trip():
    password_hash = hash_password("password123")

    assert password_hash != "password123"
    assert verify_password("password123", password_hash) is True
    assert verify_password("wrong-password", password_hash) is False


def test_access_token_round_trip():
    token = create_access_token(
        user_id=42,
        email="User@Example.com",
        secret_key="secret",
        expires_minutes=10,
    )

    payload = decode_access_token(token, "secret")

    assert payload["sub"] == "42"
    assert payload["email"] == "user@example.com"


def test_access_token_rejects_wrong_secret():
    token = create_access_token(
        user_id=42,
        email="user@example.com",
        secret_key="secret",
        expires_minutes=10,
    )

    with pytest.raises(ValueError, match="signature"):
        decode_access_token(token, "different-secret")


def test_expired_access_token_is_rejected():
    token = create_access_token(
        user_id=42,
        email="user@example.com",
        secret_key="secret",
        expires_minutes=-1,
    )

    with pytest.raises(ValueError, match="expired"):
        decode_access_token(token, "secret")


def test_normalize_email_strips_and_lowercases():
    assert normalize_email("  User@Example.COM ") == "user@example.com"
