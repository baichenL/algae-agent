from __future__ import annotations

import os
import time

from app.services.agent_runtime.contracts_v2 import (
    EffectClass,
    RuntimeBudgets,
    SAFE_AGENT_EFFECTS,
    SafetyEnvelope,
)
from app.services.agent_runtime.state import RuntimeRequestContext


POLICY_VERSION = "agent-safety-envelope-v2"


def agent_tool_loop_mode() -> str:
    value = os.getenv("AGENT_TOOL_LOOP_MODE", "off").strip().lower()
    return value if value in {"off", "shadow", "canary", "full"} else "off"


def _role_effects(role: str) -> set[EffectClass]:
    normalized = role.strip().lower()
    if normalized == "viewer":
        return {EffectClass.READ}
    if normalized in {"scientist", "approver"}:
        return set(SAFE_AGENT_EFFECTS)
    return {EffectClass.READ, EffectClass.CONTROL_WRITE}


def build_safety_envelope(
    context: RuntimeRequestContext,
    *,
    control_flow: str,
    ttl_seconds: int | None = None,
) -> SafetyEnvelope:
    if not context.owner or not context.workspace_id or not context.role:
        raise ValueError("server_derived_runtime_context_required")
    now = time.time()
    ttl = max(30, int(ttl_seconds or os.getenv("AGENT_SAFETY_ENVELOPE_TTL_SECONDS", "300")))
    allowed = _role_effects(context.role)
    return SafetyEnvelope(
        principal_id=context.owner,
        workspace_id=context.workspace_id,
        role=context.role.strip().lower(),
        control_flow=control_flow,
        allowed_effect_classes=tuple(sorted(item.value for item in allowed)),
        denied_capabilities=(
            EffectClass.DOMAIN_FACT_COMMIT.value,
            EffectClass.EXTERNAL_ACTUATION.value,
        ),
        policy_version=POLICY_VERSION,
        budgets=RuntimeBudgets.from_env(),
        created_at_epoch=now,
        expires_at_epoch=now + ttl,
    )
