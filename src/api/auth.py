"""
Phase 6: demo login + JWT session tokens.
SECURITY: /chat derives role from this signed token — never from a
client-supplied role field. See note above.
"""
import os
import time
import jwt
from fastapi import HTTPException, Header

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET is not set. Add a strong random secret to your .env file "
        '(generate one via: python -c "import secrets; print(secrets.token_hex(32))").'
    )
JWT_ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 8 * 60 * 60

DEMO_USERS = {
    "dr.mehta": {"password": "doctor", "role": "doctor"},
    "nurse.priya": {"password": "nurse", "role": "nurse"},
    "billing.ravi": {"password": "billing_executive", "role": "billing_executive"},
    "tech.anand": {"password": "technician", "role": "technician"},
    "admin.sys": {"password": "admin", "role": "admin"},
}


def authenticate(username: str, password: str) -> str:
    user = DEMO_USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return user["role"]


def create_token(username: str, role: str) -> str:
    payload = {"username": username, "role": role, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token")


def get_current_role(authorization: str = Header(...)) -> str:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    payload = decode_token(authorization.removeprefix("Bearer "))
    return payload["role"]