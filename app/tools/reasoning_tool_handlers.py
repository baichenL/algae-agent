from __future__ import annotations

import os
from typing import Any

from app.core.db.agent_artifacts_v2 import (
    get_agent_artifact,
    list_agent_artifacts,
    save_agent_artifact,
)
from app.tools.registry_core import register_agent_tool


HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string", "minLength": 1},
                    "claim": {"type": "string", "minLength": 1},
                    "status": {
                        "type": "string",
                        "enum": ["active", "supported", "weakened", "rejected"],
                    },
                    "supporting_evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "contradicting_evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                    "next_best_test": {"type": ["string", "null"]},
                },
                "required": ["hypothesis_id", "claim", "status", "confidence"],
                "additionalProperties": False,
            },
        },
        "decision_summary": {"type": "string"},
    },
    "required": ["hypotheses", "decision_summary"],
    "additionalProperties": False,
}


CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "minLength": 1},
                "version": {"type": "integer", "minimum": 1},
                "design": {"type": "object"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "hypothesis_refs": {"type": "array", "items": {"type": "string"}},
                "validation_result": {"type": "object"},
                "simulation_result": {"type": "object"},
                "risk_summary": {"type": "object"},
                "uncertainties": {"type": "array", "items": {"type": "string"}},
                "status": {
                    "type": "string",
                    "enum": ["draft", "validated", "simulated", "rejected", "preferred"],
                },
            },
            "required": ["candidate_id", "version", "design", "status"],
            "additionalProperties": False,
        },
        "comparison_summary": {"type": "string"},
    },
    "required": ["candidate", "comparison_summary"],
    "additionalProperties": False,
}


PLAN_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "patch_id": {"type": "string", "minLength": 1},
        "candidate_id": {"type": "string", "minLength": 1},
        "base_version": {"type": "integer", "minimum": 1},
        "new_version": {"type": "integer", "minimum": 2},
        "reason": {"type": "string", "minLength": 1},
        "changes": {"type": "object"},
        "addresses_error_codes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "patch_id",
        "candidate_id",
        "base_version",
        "new_version",
        "reason",
        "changes",
    ],
    "additionalProperties": False,
}


def _result(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = {"action": action, "status": "success", **payload}
    return {**response, "response_payload": response}


def _payload_summary(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, (list, dict)):
            return {"type": type(value).__name__, "count": len(value)}
        return value
    if isinstance(value, list):
        return {
            "type": "list",
            "count": len(value),
            "sample": [_payload_summary(item, depth=depth + 1) for item in value[:3]],
        }
    if isinstance(value, dict):
        return {
            key: _payload_summary(item, depth=depth + 1)
            for key, item in value.items()
        }
    return value


@register_agent_tool(
    name="agent_artifact_read",
    schema={
        "type": "object",
        "properties": {
            "artifact_ref": {"type": "string", "minLength": 1},
            "include_payload": {"type": "boolean"},
        },
        "required": ["artifact_ref"],
        "additionalProperties": False,
    },
    description="Read a previously truncated artifact from this same agent run. Summary mode is the default.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime"],
    exposed_to_llm=True,
    executor_kind="read",
    result_authority="tool_derived",
    audit_event_type="agent_artifact_read",
)
def read_agent_artifact(args: dict[str, Any]) -> dict[str, Any]:
    artifact = get_agent_artifact(str(args["artifact_ref"]))
    run_id = args.get("agent_run_id")
    if not artifact or (run_id and artifact.get("agent_run_id") != run_id):
        payload = {
            "action": "agent_artifact_read",
            "status": "not_found",
            "artifact_ref": args["artifact_ref"],
        }
        return {**payload, "response_payload": payload}
    original = artifact.get("payload") or {}
    payload = {
        "artifact_ref": artifact["artifact_id"],
        "artifact_type": artifact["artifact_type"],
        "version": artifact["version"],
        "payload_hash": artifact["payload_hash"],
        "summary": _payload_summary(original),
    }
    if args.get("include_payload"):
        payload["payload"] = original
    return _result("agent_artifact_read", payload)


@register_agent_tool(
    name="hypothesis_ledger_update",
    schema=HYPOTHESIS_SCHEMA,
    risk_level="none",
    effect_kind="compute",
    effect_class="control_write",
    side_effect="control_state",
    allowed_callers=["chat_runtime"],
    exposed_to_llm=True,
    executor_kind="compute",
    audit_event_type="hypothesis_updated",
)
def update_hypothesis_ledger(args: dict[str, Any]) -> dict[str, Any]:
    artifact = save_agent_artifact(
        agent_run_id=args.get("agent_run_id"),
        artifact_type="hypothesis_ledger",
        payload={
            "hypotheses": args["hypotheses"],
            "decision_summary": args["decision_summary"],
        },
    )
    return _result(
        "hypothesis_ledger_update",
        {
            "hypotheses": args["hypotheses"],
            "decision_summary": args["decision_summary"],
            "artifact_ref": artifact["artifact_id"],
            "version": artifact["version"],
        },
    )


@register_agent_tool(
    name="candidate_plan_record",
    schema=CANDIDATE_SCHEMA,
    risk_level="none",
    effect_kind="compute",
    effect_class="artifact_write",
    side_effect="scientific_artifacts",
    allowed_callers=["chat_runtime"],
    exposed_to_llm=True,
    executor_kind="compute",
    audit_event_type="candidate_compared",
)
def record_candidate_plan(args: dict[str, Any]) -> dict[str, Any]:
    run_id = args.get("agent_run_id")
    existing = list_agent_artifacts(agent_run_id=run_id, artifact_type="candidate_plan")
    maximum = max(1, int(os.getenv("AGENT_MAX_CANDIDATES", "3")))
    candidate_id = str(args["candidate"]["candidate_id"])
    existing_ids = {
        str((item.get("payload") or {}).get("candidate", {}).get("candidate_id"))
        for item in existing
    }
    if candidate_id not in existing_ids and len(existing_ids) >= maximum:
        payload = {
            "action": "candidate_plan_record",
            "status": "error",
            "error_code": "candidate_budget_exhausted",
            "candidate_budget": maximum,
        }
        return {**payload, "response_payload": payload}
    artifact = save_agent_artifact(
        agent_run_id=run_id,
        artifact_type="candidate_plan",
        payload={
            "candidate": args["candidate"],
            "comparison_summary": args["comparison_summary"],
        },
    )
    return _result(
        "candidate_plan_record",
        {
            "candidate": args["candidate"],
            "comparison_summary": args["comparison_summary"],
            "artifact_ref": artifact["artifact_id"],
            "version": artifact["version"],
        },
    )


@register_agent_tool(
    name="plan_patch_record",
    schema=PLAN_PATCH_SCHEMA,
    risk_level="none",
    effect_kind="compute",
    effect_class="artifact_write",
    side_effect="scientific_artifacts",
    allowed_callers=["chat_runtime"],
    exposed_to_llm=True,
    executor_kind="compute",
    audit_event_type="plan_patch_created",
)
def record_plan_patch(args: dict[str, Any]) -> dict[str, Any]:
    run_id = args.get("agent_run_id")
    existing = list_agent_artifacts(agent_run_id=run_id, artifact_type="plan_patch")
    maximum = max(1, int(os.getenv("AGENT_MAX_PLAN_PATCHES", "3")))
    patch_ids = {
        str((item.get("payload") or {}).get("patch_id"))
        for item in existing
    }
    if args["patch_id"] not in patch_ids and len(patch_ids) >= maximum:
        payload = {
            "action": "plan_patch_record",
            "status": "error",
            "error_code": "plan_patch_budget_exhausted",
            "plan_patch_budget": maximum,
        }
        return {**payload, "response_payload": payload}
    artifact = save_agent_artifact(
        agent_run_id=run_id,
        artifact_type="plan_patch",
        payload={
            key: value
            for key, value in args.items()
            if key not in {"agent_run_id", "created_by_tool_call_id"}
        },
    )
    return _result(
        "plan_patch_record",
        {
            "patch": {
                key: value
                for key, value in args.items()
                if key not in {"agent_run_id", "created_by_tool_call_id"}
            },
            "artifact_ref": artifact["artifact_id"],
            "version": artifact["version"],
        },
    )
