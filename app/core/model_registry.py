from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import client


ModelRole = Literal[
    "chat",
    "router",
    "agent",
    "replan",
    "rag_answer",
    "reflection",
    "memory",
]


@dataclass(frozen=True)
class ModelRoleConfig:
    role: ModelRole
    model: str
    source: str


_ROLE_ENV: dict[ModelRole, tuple[str, ...]] = {
    "chat": ("ALGAE_MODEL_CHAT",),
    "router": ("ALGAE_MODEL_ROUTER", "HYBRID_ROUTER_LLM_MODEL"),
    "agent": ("ALGAE_MODEL_AGENT", "AGENT_TOOL_MODEL"),
    "replan": ("ALGAE_MODEL_REPLAN", "AGENT_REPLAN_LLM_MODEL"),
    "rag_answer": ("ALGAE_MODEL_RAG_ANSWER",),
    "reflection": ("ALGAE_MODEL_REFLECTION",),
    "memory": ("ALGAE_MODEL_MEMORY", "USER_MEMORY_EXTRACTION_MODEL"),
}

_ROLE_DEFAULT: dict[ModelRole, str] = {
    "chat": "deepseek-v4-flash",
    "router": "deepseek-v4-flash",
    "agent": "deepseek-v4-pro",
    "replan": "deepseek-v4-pro",
    "rag_answer": "deepseek-v4-flash",
    "reflection": "deepseek-v4-pro",
    "memory": "deepseek-v4-flash",
}

_LEGACY_ENV = {
    "HYBRID_ROUTER_LLM_MODEL",
    "AGENT_TOOL_MODEL",
    "AGENT_REPLAN_LLM_MODEL",
    "USER_MEMORY_EXTRACTION_MODEL",
}

_probe_lock = threading.Lock()
_probe_status: dict[str, dict[str, Any]] = {}


def model_config(role: ModelRole) -> ModelRoleConfig:
    for name in _ROLE_ENV[role]:
        value = str(os.getenv(name) or "").strip()
        if not value:
            continue
        if name in _LEGACY_ENV:
            warnings.warn(
                f"{name} is deprecated; use {_ROLE_ENV[role][0]} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return ModelRoleConfig(role=role, model=value, source=name)
    return ModelRoleConfig(role=role, model=_ROLE_DEFAULT[role], source="default")


def model_name(role: ModelRole) -> str:
    return model_config(role).model


def configured_model_status() -> dict[str, Any]:
    roles = {
        role: {
            "status": "configured" if model_config(role).model else "unavailable",
            "model": model_config(role).model,
            "source": model_config(role).source,
        }
        for role in _ROLE_ENV
    }
    required_ready = all(roles[role]["status"] == "configured" for role in ("chat", "agent", "replan"))
    return {"status": "ready" if required_ready else "unavailable", "roles": roles}


def probe_model_capabilities(*, force: bool = False) -> dict[str, Any]:
    """Probe unique configured completion models once and cache a safe status.

    Tests and offline deployments can disable remote probing with
    MODEL_CAPABILITY_PROBE_ENABLED=false. Readiness still reports the resolved
    role configuration in that mode.
    """

    enabled = str(os.getenv("MODEL_CAPABILITY_PROBE_ENABLED", "true")).casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return {**configured_model_status(), "probe": "disabled"}

    with _probe_lock:
        if _probe_status and not force:
            return dict(_probe_status)
        unique_models = sorted({model_name(role) for role in _ROLE_ENV})
        models: dict[str, dict[str, Any]] = {}
        for model in unique_models:
            try:
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "health"}],
                    temperature=0,
                    max_tokens=1,
                )
                models[model] = {"status": "ready"}
            except Exception as exc:
                failure = classify_provider_error(exc)
                models[model] = {
                    "status": "unavailable",
                    "code": failure["code"],
                    "retryable": failure["retryable"],
                }
        roles = {
            role: {
                "model": model_name(role),
                **models.get(model_name(role), {"status": "unavailable", "code": "model_probe_missing"}),
            }
            for role in _ROLE_ENV
        }
        required_ready = all(roles[role]["status"] == "ready" for role in ("chat", "agent", "replan"))
        _probe_status.clear()
        _probe_status.update(
            {
                "status": "ready" if required_ready else "unavailable",
                "probe": "completed",
                "roles": roles,
            }
        )
        return dict(_probe_status)


def model_runtime_status() -> dict[str, Any]:
    if _probe_status:
        return dict(_probe_status)
    return {**configured_model_status(), "probe": "not_run"}


def reset_model_probe_status() -> None:
    with _probe_lock:
        _probe_status.clear()


def classify_provider_error(exc: Exception) -> dict[str, Any]:
    from app.core.provider_errors import safe_provider_error

    return safe_provider_error(exc)
