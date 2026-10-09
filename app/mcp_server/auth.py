"""Bearer tokens for the MCP server: identity travels in the token, never in tool arguments.

Token = base64url(JSON {sub: employee_id, role, exp}) + "." + base64url(HMAC-SHA256(secret, payload)).
A tiny signed token (JWT-like, without a JWT dependency): the API mints one per agent request
(TOOLS_VIA_MCP=true), and `python -m app.mcp_server token E1004` mints one for an MCP client.
"""

import base64
import hashlib
import hmac
import json
import time

from mcp.server.auth.provider import AccessToken

from app.config import get_settings

ROLES = ("employee", "hr", "it_admin")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload: str) -> str:
    return _b64(hmac.new(get_settings().mcp_auth_secret.encode(), payload.encode(), hashlib.sha256).digest())


def mint_token(employee_id: str, role: str, ttl_s: int = 300) -> str:
    if role not in ROLES:
        raise ValueError(f"unknown role {role}")
    payload = _b64(json.dumps({"sub": employee_id, "role": role, "exp": int(time.time()) + ttl_s}).encode())
    return f"{payload}.{_sign(payload)}"


def verify(token: str) -> dict | None:
    """Claims if the signature is valid and the token hasn't expired, else None."""
    try:
        payload, sig = token.split(".", 1)
        if not hmac.compare_digest(sig, _sign(payload)):  # constant-time comparison
            return None
        claims = json.loads(_unb64(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if claims.get("exp", 0) < time.time() or claims.get("role") not in ROLES or not claims.get("sub"):
        return None
    return claims


class HmacTokenVerifier:
    """mcp TokenVerifier: the SDK's bearer middleware calls this for every HTTP request."""

    async def verify_token(self, token: str) -> AccessToken | None:
        claims = verify(token)
        if claims is None:
            return None
        return AccessToken(
            token=token,
            client_id=claims["sub"],
            subject=claims["sub"],
            scopes=[f"role:{claims['role']}"],
            expires_at=claims["exp"],
            claims=claims,
        )
