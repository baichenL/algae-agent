from __future__ import annotations

from typing import Any

from app.services.agent_runtime.contracts_v2 import (
    EffectClass,
    ResolvedTool,
    SafetyEnvelope,
)
from app.services.context.budget import estimate_tokens
from app.tools.registry import AGENT_TOOL_REGISTRY


_LEGACY_EFFECT_MAP = {
    "read": EffectClass.READ,
    "compute": EffectClass.COMPUTE,
    "draft": EffectClass.ARTIFACT_WRITE,
    "propose": EffectClass.PROPOSAL_WRITE,
    "execute": EffectClass.DOMAIN_FACT_COMMIT,
}

_SERVER_OWNED_ARGUMENTS = {
    "workspace_id",
    "principal_id",
    "owner",
    "role",
    "caller",
    "approval_status",
    "approval_version",
    "session_id",
    "agent_run_id",
    "graph_thread_id",
    "execution_idempotency_key",
    "domain_dedupe_key",
    "created_by_tool_call_id",
}


def normalized_effect_class(metadata: dict[str, Any]) -> EffectClass:
    value = str(metadata.get("effect_class") or "").strip().lower()
    if value:
        return EffectClass(value)
    legacy = str(metadata.get("effect_kind") or "read").strip().lower()
    if metadata.get("side_effect") == "scientific_artifacts":
        return EffectClass.ARTIFACT_WRITE
    return _LEGACY_EFFECT_MAP.get(legacy, EffectClass.READ)


def _normalized_schema(name: str, definition: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    schema = definition.get("schema") or {}
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        function = schema["function"]
        parameters = dict(function.get("parameters") or {})
        description = str(function.get("description") or name)
    else:
        parameters = dict(schema)
        description = str(definition.get("description") or name.replace("_", " "))
    properties = dict(parameters.get("properties") or {})
    parameters["properties"] = {
        key: value for key, value in properties.items() if key not in _SERVER_OWNED_ARGUMENTS
    }
    parameters["required"] = [
        key for key in parameters.get("required") or [] if key not in _SERVER_OWNED_ARGUMENTS
    ]
    parameters.setdefault("additionalProperties", False)
    return description, parameters


def resolve_tools(envelope: SafetyEnvelope) -> list[ResolvedTool]:
    if envelope.is_expired():
        raise PermissionError("safety_envelope_expired")
    resolved: list[ResolvedTool] = []
    for name, definition in sorted(AGENT_TOOL_REGISTRY.items()):
        metadata = definition.get("metadata") or definition
        if not bool(metadata.get("exposed_to_llm")):
            continue
        effect = normalized_effect_class(metadata)
        if not envelope.allows(effect):
            continue
        allowed_roles = tuple(str(item).lower() for item in metadata.get("allowed_roles") or ())
        if allowed_roles and envelope.role not in allowed_roles:
            continue
        description, input_schema = _normalized_schema(name, definition)
        resolved.append(
            ResolvedTool(
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=dict(metadata.get("output_schema") or {"type": "object"}),
                effect_class=effect.value,
                risk_level=str(metadata.get("risk_level") or "low"),
                requires_approval=bool(metadata.get("requires_approval")),
                executor_kind=str(metadata.get("executor_kind") or effect.value),
                parallel_safe=bool(metadata.get("parallel_safe", effect == EffectClass.READ)),
                timeout_seconds=max(1, int(metadata.get("timeout_seconds") or 30)),
                result_authority=str(metadata.get("result_authority") or "tool_derived"),
            )
        )
    schemas = [item.to_openai_tool() for item in resolved]
    estimated = estimate_tokens(schemas)
    if estimated > envelope.budgets.tool_schema_tokens:
        raise RuntimeError(
            f"tool_schema_budget_exceeded:{estimated}>{envelope.budgets.tool_schema_tokens}"
        )
    return resolved
