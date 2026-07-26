# app/tools/registry_core.py
# 负责注册和管理 Agent 工具
from typing import Any, Dict

AGENT_TOOL_REGISTRY = {}

DEFAULT_TOOL_METADATA: Dict[str, Any] = {
    "risk_level": "low",
    "effect_kind": "read",
    "effect_class": None,
    "side_effect": "none",
    "requires_approval": False,
    "allowed_callers": ["chat_runtime"],
    "allowed_roles": [],
    "idempotency_fields": [],
    "audit_event_type": None,
    # New tools are private until an owner explicitly reviews their schema,
    # effect classification, and executor capability.
    "exposed_to_llm": False,
    "executor_kind": None,
    "parallel_safe": False,
    "timeout_seconds": 30,
    "result_authority": "tool_derived",
    "output_schema": {"type": "object"},
}


def normalize_tool_metadata(name: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    raw = dict(metadata or {})
    normalized = {**DEFAULT_TOOL_METADATA, **raw}
    if normalized["audit_event_type"] is None:
        normalized["audit_event_type"] = name
    if "requires_explicit_confirmation" in raw and "requires_approval" not in raw:
        normalized["requires_approval"] = bool(raw["requires_explicit_confirmation"])
    normalized["allowed_callers"] = list(normalized.get("allowed_callers") or [])
    normalized["allowed_roles"] = list(normalized.get("allowed_roles") or [])
    normalized["idempotency_fields"] = list(normalized.get("idempotency_fields") or [])
    return normalized


def register_agent_tool(name: str, schema: Dict[str, Any], **metadata: Any):
    def decorator(func):
        tool_metadata = normalize_tool_metadata(name, metadata)
        AGENT_TOOL_REGISTRY[name] = {
            "schema": schema,
            "handler": func,
            **tool_metadata,
            "metadata": tool_metadata,
        }
        return func
    return decorator
