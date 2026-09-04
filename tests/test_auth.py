"""
Unit tests for JWT token creation/validation — no live server, no Qdrant,
no Groq needed. Runs in well under a second.
"""
from dotenv import load_dotenv
load_dotenv()

import time
import jwt
import pytest
from fastapi import HTTPException

from src.api.auth import create_token, decode_token, authenticate, JWT_SECRET, JWT_ALGORITHM


def test_create_and_decode_roundtrip():
    token = create_token("dr.mehta", "doctor")
    payload = decode_token(token)
    assert payload["username"] == "dr.mehta"
    assert payload["role"] == "doctor"


def test_decode_rejects_tampered_token():
    token = create_token("dr.mehta", "doctor")
    tampered = token[:-5] + "aaaaa"  # corrupt the signature
    with pytest.raises(HTTPException) as exc_info:
        decode_token(tampered)
    assert exc_info.value.status_code == 401


def test_decode_rejects_wrong_secret():
    # A token signed with a DIFFERENT secret must be rejected - this is
    # what stops a forged token from ever being accepted.
    forged = jwt.encode({"username": "x", "role": "admin"}, "wrong-secret", algorithm=JWT_ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        decode_token(forged)
    assert exc_info.value.status_code == 401


def test_decode_rejects_expired_token():
    expired_payload = {"username": "dr.mehta", "role": "doctor", "exp": int(time.time()) - 10}
    expired_token = jwt.encode(expired_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    with pytest.raises(HTTPException) as exc_info:
        decode_token(expired_token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_authenticate_correct_credentials():
    role = authenticate("nurse.priya", "nurse")
    assert role == "nurse"


def test_authenticate_wrong_password():
    with pytest.raises(HTTPException) as exc_info:
        authenticate("nurse.priya", "wrong-password")
    assert exc_info.value.status_code == 401


def test_authenticate_unknown_username():
    with pytest.raises(HTTPException) as exc_info:
        authenticate("nobody.here", "whatever")
    assert exc_info.value.status_code == 401