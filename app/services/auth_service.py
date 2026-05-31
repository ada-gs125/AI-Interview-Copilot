"""Password hashing and minimal HS256 JWT helpers for auth routes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any


def normalize_email(email: str) -> str:
    # Store and compare email addresses in one canonical form.
    return email.strip().lower()


def hash_password(password: str) -> str:
    # PBKDF2 with a per-password random salt; no external auth dependency needed.
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return f"pbkdf2_sha256${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, password_hash: str) -> bool:
    # Recompute the PBKDF2 digest and compare in constant time.
    try:
        algorithm, salt_b64, digest_b64 = password_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False

    salt = _b64decode(salt_b64)
    expected = _b64decode(digest_b64)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return hmac.compare_digest(actual, expected)


def create_access_token(
    *,
    user_id: int,
    email: str,
    secret_key: str,
    expires_minutes: int,
) -> str:
    # Build a compact JWT-like token signed with HMAC-SHA256.
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": normalize_email(email),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_json_b64(header)}.{_json_b64(payload)}"
    signature = _sign(signing_input, secret_key)
    return f"{signing_input}.{signature}"


def decode_access_token(token: str, secret_key: str) -> dict[str, Any]:
    # Validate token structure, signature, algorithm, type, expiry, and subject.
    try:
        header_b64, payload_b64, signature = token.split(".", 2)
    except ValueError as exc:
        raise ValueError("Malformed access token.") from exc

    # Verify signature first — reject any tampered token before parsing its content.
    signing_input = f"{header_b64}.{payload_b64}"
    expected_signature = _sign(signing_input, secret_key)
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid access token signature.")

    header = json.loads(_b64decode(header_b64))
    if header.get("alg") != "HS256":
        raise ValueError("Unsupported access token algorithm.")
    if header.get("typ") != "JWT":
        raise ValueError("Unsupported access token type.")

    payload = json.loads(_b64decode(payload_b64))
    exp = payload.get("exp")
    if not isinstance(exp, int) or exp < int(datetime.now(timezone.utc).timestamp()):
        raise ValueError("Access token has expired.")

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub.isdigit():
        raise ValueError("Access token has invalid subject claim.")

    return payload


def _sign(value: str, secret_key: str) -> str:
    digest = hmac.new(secret_key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    return _b64encode(digest)


def _json_b64(value: dict[str, Any]) -> str:
    return _b64encode(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    try:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)
    except Exception as exc:
        raise ValueError("Malformed base64 encoding in access token.") from exc
