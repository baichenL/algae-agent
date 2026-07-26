from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from typing import Any

from jsonschema import Draft202012Validator

from app.core.db.agent_artifacts_v2 import save_agent_artifact
from app.core.effects import EffectClass, stable_hash
from app.services.agent_runtime.contracts_v2 import (
    EvidenceRef,
    ModelToolCall,
    ResolvedTool,
    SafetyEnvelope,
    ToolObservation,
)
from app.services.context.budget import estimate_tokens
from app.tools.executor import ToolExecutionContext, execute_registered_tool


SERVER_OWNED_ARGUMENTS = {
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


def _error_observation(
    call: ModelToolCall,
    *,
    effect_class: str = EffectClass.READ.value,
    authority: str = "control_state",
    code: str,
    details: dict[str, Any],
    repairs: list[str] | None = None,
    retryable: bool = False,
) -> ToolObservation:
    return ToolObservation(
        observation_id=str(uuid.uuid4()),
        tool_call_id=call.tool_call_id,
        tool_name=call.name,
        status="error",
        effect_class=effect_class,
        authority=authority,
        error_code=code,
        error_details=details,
        suggested_repairs=tuple(repairs or ()),
        retryable=retryable,
    )


def _validate_call(
    call: ModelToolCall,
    resolved: dict[str, ResolvedTool],
    envelope: SafetyEnvelope,
) -> tuple[ResolvedTool | None, ToolObservation | None]:
    tool = resolved.get(call.name)
    if tool is None:
        return None, _error_observation(
            call,
            code="tool_not_resolved",
            details={"tool_name": call.name},
            repairs=["Choose a tool from the current resolved tool schemas."],
        )
    if envelope.is_expired():
        return None, _error_observation(
            call,
            effect_class=tool.effect_class,
            authority=tool.result_authority,
            code="safety_envelope_expired",
            details={},
        )
    if not envelope.allows(tool.effect_class):
        return None, _error_observation(
            call,
            effect_class=tool.effect_class,
            authority=tool.result_authority,
            code="policy_denied",
            details={"effect_class": tool.effect_class},
        )
    forbidden = sorted(SERVER_OWNED_ARGUMENTS.intersection(call.arguments))
    if forbidden:
        return None, _error_observation(
            call,
            effect_class=tool.effect_class,
            authority=tool.result_authority,
            code="server_owned_argument",
            details={"fields": forbidden},
            repairs=["Remove server-owned identity, workspace, approval, and trace fields."],
        )
    errors = sorted(
        Draft202012Validator(tool.input_schema).iter_errors(call.arguments),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        field_errors = [
            {
                "path": ".".join(str(part) for part in error.absolute_path) or "$",
                "message": error.message,
                "validator": error.validator,
            }
            for error in errors
        ]
        return None, _error_observation(
            call,
            effect_class=tool.effect_class,
            authority=tool.result_authority,
            code="invalid_arguments",
            details={"field_errors": field_errors},
            repairs=[f"Correct {item['path']}: {item['message']}" for item in field_errors],
        )
    return tool, None


def _result_data(result: dict[str, Any]) -> dict[str, Any]:
    payload = result.get("response_payload")
    if isinstance(payload, dict):
        return payload
    return {key: value for key, value in result.items() if key != "memory_text"}


def _normalize_observation(
    call: ModelToolCall,
    tool: ResolvedTool,
    result: dict[str, Any],
    *,
    agent_run_id: str,
    duration_ms: int,
    inline_token_budget: int,
) -> ToolObservation:
    data = _result_data(result)
    status = str(result.get("status") or data.get("status") or "success")
    evidence_refs: tuple[EvidenceRef, ...] = ()
    if status in {"success", "pending", "not_found"}:
        evidence_refs = (
            EvidenceRef(
                ref_id=f"evidence:{stable_hash({'tool': call.name, 'data': data})[:24]}",
                authority=tool.result_authority,
                source=call.name,
                resource_version=str(data.get("resource_version")) if data.get("resource_version") else None,
                summary=str(data.get("message") or data.get("msg") or status)[:300],
            ),
        )
    pending_id = data.get("pending_id") or result.get("pending_id")
    try:
        pending_id = int(pending_id) if pending_id is not None else None
    except (TypeError, ValueError):
        pending_id = None
    error_code = None
    error_details: dict[str, Any] = {}
    suggested_repairs: tuple[str, ...] = ()
    retryable = bool(result.get("retryable") or data.get("retryable"))
    if status in {
        "error",
        "blocked",
        "needs_more_info",
        "simulation_failed",
        "proposal_not_ready",
        "conflicted",
        "target_mismatch",
    }:
        error_code = str(
            result.get("error_code")
            or data.get("error_code")
            or ("missing_information" if status == "needs_more_info" else status)
        )
        error_details = {
            "message": result.get("message") or data.get("message"),
            "missing_fields": data.get("missing_fields") or result.get("missing_fields") or [],
            "error_event_id": result.get("error_event_id") or data.get("error_event_id"),
            "validation": data.get("validation"),
            "simulation": data.get("simulation"),
            "conflicting_evidence": data.get("conflicting_evidence") or [],
        }
        suggested_repairs = tuple(
            data.get("questions")
            or result.get("questions")
            or data.get("suggested_next_investigations")
            or data.get("repairable_fields")
            or ()
        )
    truncated = estimate_tokens(data) > inline_token_budget
    artifact_ref = None
    if truncated:
        artifact = save_agent_artifact(
            agent_run_id=agent_run_id,
            artifact_type="tool_observation_payload",
            payload={
                "tool_call_id": call.tool_call_id,
                "tool_name": call.name,
                "data": data,
            },
        )
        artifact_ref = artifact["artifact_id"]
        data = {
            "status": status,
            "message": data.get("message") or data.get("msg"),
            "top_level_fields": sorted(data),
            "artifact_ref": artifact_ref,
        }
    return ToolObservation(
        observation_id=str(uuid.uuid4()),
        tool_call_id=call.tool_call_id,
        tool_name=call.name,
        status=status,
        effect_class=tool.effect_class,
        authority=tool.result_authority,
        data=data,
        evidence_refs=evidence_refs,
        error_code=error_code,
        error_details=error_details,
        retryable=retryable,
        suggested_repairs=suggested_repairs,
        pending_id=pending_id,
        resource_versions=dict(data.get("resource_versions") or {}),
        truncated=truncated,
        artifact_ref=artifact_ref,
        duration_ms=duration_ms,
    )


async def _execute_one(
    call: ModelToolCall,
    tool: ResolvedTool,
    envelope: SafetyEnvelope,
    *,
    session_id: str,
    agent_run_id: str,
) -> ToolObservation:
    started = time.monotonic()
    context = ToolExecutionContext(
        caller="chat_runtime",
        session_id=session_id,
        agent_run_id=agent_run_id,
        source="agent_tool_loop_v2",
        principal_id=envelope.principal_id,
        workspace_id=envelope.workspace_id,
        role=envelope.role,
        tool_call_id=call.tool_call_id,
    )
    attempts = 2 if tool.effect_class in {EffectClass.READ.value, EffectClass.COMPUTE.value} else 1
    result: dict[str, Any] = {}
    for attempt in range(attempts):
        result = await execute_registered_tool(call.name, call.arguments, context=context)
        retryable = bool(result.get("retryable") or (result.get("response_payload") or {}).get("retryable"))
        if str(result.get("status")) not in {"error"} or not retryable or attempt + 1 >= attempts:
            break
        await asyncio.sleep(0)
    return _normalize_observation(
        call,
        tool,
        result,
        agent_run_id=agent_run_id,
        duration_ms=int((time.monotonic() - started) * 1000),
        inline_token_budget=envelope.budgets.observation_inline_tokens,
    )


async def execute_tool_calls(
    calls: tuple[ModelToolCall, ...],
    tools: list[ResolvedTool],
    envelope: SafetyEnvelope,
    *,
    session_id: str,
    agent_run_id: str,
) -> list[ToolObservation]:
    resolved = {item.name: item for item in tools}
    observations: list[ToolObservation] = []
    executable: list[tuple[ModelToolCall, ResolvedTool]] = []
    for call in calls:
        tool, error = _validate_call(call, resolved, envelope)
        if error is not None:
            observations.append(error)
        elif tool is not None:
            executable.append((call, tool))

    safe_parallel = [
        item for item in executable
        if item[1].parallel_safe and item[1].effect_class != EffectClass.PROPOSAL_WRITE.value
    ]
    serial = [item for item in executable if item not in safe_parallel]
    if safe_parallel:
        observations.extend(
            await asyncio.gather(
                *(
                    _execute_one(
                        call,
                        tool,
                        envelope,
                        session_id=session_id,
                        agent_run_id=agent_run_id,
                    )
                    for call, tool in safe_parallel
                )
            )
        )
    for call, tool in serial:
        observations.append(
            await _execute_one(
                call,
                tool,
                envelope,
                session_id=session_id,
                agent_run_id=agent_run_id,
            )
        )
    call_order = {call.tool_call_id: index for index, call in enumerate(calls)}
    observations.sort(key=lambda item: call_order.get(item.tool_call_id, len(call_order)))
    return observations


def observation_dicts(observations: list[ToolObservation]) -> list[dict[str, Any]]:
    return [item.to_model_payload() for item in observations]
