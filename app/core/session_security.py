from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass

from fastapi import Cookie, Header, HTTPException, Request, status

from app.core.security import ApiPrincipal, ROLE_RANK, principal_for_api_key


SESSION_COOKIE = "algae_session"
SESSION_TTL_SECONDS = 12 * 60 * 60


@dataclass
class BrowserSession:
    token: str
    csrf_token: str
    principal: ApiPrincipal
    expires_at: float


_sessions: dict[str, BrowserSession] = {}


def create_browser_session(api_key: str | None) -> BrowserSession:
    if os.getenv("ALGAE_AUTH_MODE", "required").casefold() == "test":
        return _test_session()
    principal = principal_for_api_key(api_key)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A configured API key is required.",
        )
    session = BrowserSession(
        token=secrets.token_urlsafe(32),
        csrf_token=secrets.token_urlsafe(24),
        principal=principal,
        expires_at=time.time() + SESSION_TTL_SECONDS,
    )
    _sessions[session.token] = session
    return session


def delete_browser_session(token: str | None) -> None:
    if token:
        _sessions.pop(token, None)


def _test_session() -> BrowserSession:
    return BrowserSession(
        token="test-session",
        csrf_token="test-csrf",
        principal=ApiPrincipal(name="pytest-approver", role="approver", source="test_mode"),
        expires_at=time.time() + SESSION_TTL_SECONDS,
    )


def get_browser_session(token: str | None) -> BrowserSession | None:
    if os.getenv("ALGAE_AUTH_MODE", "required").casefold() == "test":
        return _test_session()
    if not token:
        return None
    session = _sessions.get(token)
    if not session:
        return None
    if session.expires_at <= time.time():
        _sessions.pop(token, None)
        return None
    return session


def require_session_role(minimum_role: str, *, csrf: bool = False):
    async def dependency(
        request: Request,
        algae_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_csrf_token: str | None = Header(default=None, alias="X-CSRF-Token"),
    ) -> ApiPrincipal:
        session = get_browser_session(algae_session)
        principal = session.principal if session else principal_for_api_key(x_api_key)
        if principal is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if ROLE_RANK.get(principal.role, 0) < ROLE_RANK[minimum_role]:
            raise HTTPException(status_code=403, detail=f"Role {minimum_role} or higher is required.")
        if csrf and session and x_csrf_token != session.csrf_token:
            raise HTTPException(status_code=403, detail="Invalid CSRF token.")
        return principal

    return dependency


require_session_viewer = require_session_role("viewer")
require_session_user_mutation = require_session_role("viewer", csrf=True)
require_session_scientist = require_session_role("scientist", csrf=True)
require_session_approver = require_session_role("approver", csrf=True)
