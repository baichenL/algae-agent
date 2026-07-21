from __future__ import annotations

from typing import Any

from app.services.agent_runtime.state import RuntimeRequestContext
from app.tools.registry import AGENT_TOOL_REGISTRY


def toolset_for_context(context: RuntimeRequestContext | None) -> dict[str, Any]:
    # None is the trusted legacy/internal runtime path. Public API requests
    # always receive an explicit server-derived context.
    role = (context.role if context else "scientist").strip().lower()
    allowed_effects = {
        "viewer": {"read"},
        "scientist": {"read", "draft", "propose"},
        "approver": {"read", "draft", "propose"},
    }.get(role, {"read"})
    tools: list[dict[str, Any]] = []
    for name, definition in sorted(AGENT_TOOL_REGISTRY.items()):
        metadata = definition.get("metadata") or definition
        effect = str(metadata.get("effect_kind") or "read")
        if not metadata.get("exposed_to_llm", True) or effect not in allowed_effects:
            continue
        tools.append(
            {
                "name": name,
                "effect_kind": effect,
                "risk_level": metadata.get("risk_level") or "low",
                "requires_approval": bool(metadata.get("requires_approval")),
            }
        )
    return {
        "role": role,
        "debug_mode": bool(context and context.debug_mode),
        "tools": tools,
        "tool_names": [item["name"] for item in tools],
        "approval_via_chat": False,
    }


def context_allows_tool(context: RuntimeRequestContext | None, tool_name: str) -> bool:
    return tool_name in set(toolset_for_context(context)["tool_names"])
