from __future__ import annotations

import json
import os
from dataclasses import dataclass

from fastapi import Header, HTTPException, status


ROLE_RANK = {"viewer": 1, "scientist": 2, "approver": 3}


@dataclass(frozen=True)
class ApiPrincipal:
    name: str
    role: str
    source: str = "api_key"


def _configured_keys() -> dict[str, dict[str, str]]:
    raw = os.getenv("ALGAE_API_KEYS_JSON", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for key, value in (parsed or {}).items():
        if isinstance(value, str):
            normalized[str(key)] = {"name": value, "role": value}
        elif isinstance(value, dict):
            normalized[str(key)] = {
                "name": str(value.get("name") or value.get("role") or "api-user"),
                "role": str(value.get("role") or "viewer"),
            }
    return normalized


def principal_for_api_key(api_key: str | None) -> ApiPrincipal | None:
    if os.getenv("ALGAE_AUTH_MODE", "required").casefold() == "test":
        return ApiPrincipal(name="pytest-approver", role="approver", source="test_mode")
    item = _configured_keys().get(str(api_key or ""))
    if not item or item["role"] not in ROLE_RANK:
        return None
    return ApiPrincipal(name=item["name"], role=item["role"])


# Compatibility for existing tests and extensions; new code uses the public name.
_principal_for_key = principal_for_api_key


def require_role(minimum_role: str):
    async def dependency(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> ApiPrincipal:
        principal = principal_for_api_key(x_api_key)
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A configured X-API-Key is required.",
            )
        if ROLE_RANK[principal.role] < ROLE_RANK[minimum_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {minimum_role} or higher is required.",
            )
        return principal
    return dependency


require_viewer = require_role("viewer")
require_scientist = require_role("scientist")
require_approver = require_role("approver")
