from __future__ import annotations

from typing import Any

from app.core.db import (
    agent_events,
    experiments,
    outcomes,
    pending_actions,
    scientific,
    strains,
    workflow_runs,
)
from app.models.rag_schema import RagQueryRequest
from app.services.rag.service import answer_rag_question
from app.tools.registry_core import register_agent_tool


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _result(action: str, status: str = "success", **data: Any) -> dict[str, Any]:
    payload = {"action": action, "status": status, **data}
    return {**payload, "response_payload": payload}


@register_agent_tool(
    name="strain_state_get",
    schema=_schema({"strain_id": {"type": "string", "minLength": 1}}, ["strain_id"]),
    description="Read one strain's current authoritative domain state by stable strain id.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="domain_fact",
    audit_event_type="strain_state_read",
)
def strain_state_get(args: dict[str, Any]) -> dict[str, Any]:
    item = strains.get_algae_status(args["strain_id"])
    return _result(
        "strain_state_get",
        status="success" if item else "not_found",
        strain=item,
        resource_versions={"strain_id": args["strain_id"], "generation": (item or {}).get("generation_number")},
    )


@register_agent_tool(
    name="experiment_history_get",
    schema=_schema(
        {
            "strain_id": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        ["strain_id"],
    ),
    description="Read verified experiment history and growth measurements for one strain.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="domain_fact",
    audit_event_type="experiment_history_read",
)
def experiment_history_get(args: dict[str, Any]) -> dict[str, Any]:
    items = experiments.get_recent_experiments(
        strain=args["strain_id"],
        limit=int(args.get("limit") or 20),
    )
    return _result("experiment_history_get", experiments=items, count=len(items))


@register_agent_tool(
    name="scientific_dataset_get",
    schema=_schema(
        {
            "dataset_id": {"type": "string", "minLength": 1},
            "target_strain_id": {"type": "string", "minLength": 1},
        },
        ["dataset_id"],
    ),
    description="Read a versioned scientific dataset including batches and time-series measurements.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="tool_derived",
    audit_event_type="scientific_dataset_read",
)
def scientific_dataset_get(args: dict[str, Any]) -> dict[str, Any]:
    dataset = scientific.get_dataset(args["dataset_id"])
    target_strain_id = args.get("target_strain_id")
    if (
        dataset
        and target_strain_id
        and str(dataset.get("strain_id") or "") != str(target_strain_id)
    ):
        return _result(
            "scientific_dataset_get",
            status="target_mismatch",
            dataset=None,
            target_strain_id=target_strain_id,
            dataset_strain_id=dataset.get("strain_id"),
            error_code="scientific_target_dataset_mismatch",
            retryable=False,
        )
    return _result(
        "scientific_dataset_get",
        status="success" if dataset else "not_found",
        dataset=dataset,
        resource_versions={
            "dataset_id": args["dataset_id"],
            "content_hash": (dataset or {}).get("content_hash"),
        },
    )


@register_agent_tool(
    name="scientific_run_get",
    schema=_schema({"scientific_run_id": {"type": "string", "minLength": 1}}, ["scientific_run_id"]),
    description="Read a scientific run with versioned observations, validation, simulation, and PlanPatch artifacts.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="control_state",
    audit_event_type="scientific_run_read",
)
def scientific_run_get(args: dict[str, Any]) -> dict[str, Any]:
    run = scientific.get_scientific_run(args["scientific_run_id"])
    return _result("scientific_run_get", status="success" if run else "not_found", scientific_run=run)


@register_agent_tool(
    name="knowledge_evidence_search",
    schema=_schema(
        {
            "question": {"type": "string", "minLength": 1},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            "doc_types": {"type": "array", "items": {"type": "string"}},
        },
        ["question"],
    ),
    description="Search SOP and literature evidence. Results are advisory and never override domain facts.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="rag_query_log",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="rag_advisory",
    audit_event_type="knowledge_evidence_searched",
)
def knowledge_evidence_search(args: dict[str, Any]) -> dict[str, Any]:
    response = answer_rag_question(
        RagQueryRequest(
            question=args["question"],
            top_k=int(args.get("top_k") or 5),
            doc_types=list(args.get("doc_types") or []),
        )
    )
    payload = response.model_dump(mode="json")
    return _result("knowledge_evidence_search", status=response.status, rag=payload)


@register_agent_tool(
    name="workflow_runs_list",
    schema=_schema(
        {
            "strain_id": {"type": ["string", "null"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        }
    ),
    description="Read workflow execution status, simulation state, and audit-linked run history.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="control_state",
    audit_event_type="workflow_runs_read",
)
def workflow_runs_list(args: dict[str, Any]) -> dict[str, Any]:
    items = workflow_runs.list_workflow_runs(
        strain_id=args.get("strain_id"),
        limit=int(args.get("limit") or 20),
    )
    return _result("workflow_runs_list", workflow_runs=items, count=len(items))


@register_agent_tool(
    name="agent_trace_read",
    schema=_schema({"agent_run_id_query": {"type": "string", "minLength": 1}}, ["agent_run_id_query"]),
    description="Read a prior Agent run and its structured Trace events without exposing hidden reasoning.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="control_state",
    audit_event_type="agent_trace_read",
)
def agent_trace_read(args: dict[str, Any]) -> dict[str, Any]:
    run_id = args["agent_run_id_query"]
    run = agent_events.get_agent_run(run_id)
    events = agent_events.list_agent_run_events(run_id) if run else []
    return _result(
        "agent_trace_read",
        status="success" if run else "not_found",
        run=run,
        events=events,
    )


@register_agent_tool(
    name="lab_devices_list",
    schema=_schema({}),
    description="Read registered simulation/device capabilities. No actuation capability is exposed.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="control_state",
    audit_event_type="lab_devices_read",
)
def lab_devices_list(_: dict[str, Any]) -> dict[str, Any]:
    return _result(
        "lab_devices_list",
        devices=scientific.list_lab_devices(),
        physical_execution_available=False,
    )


@register_agent_tool(
    name="state_observation_get",
    schema=_schema({"pending_id": {"type": "integer", "minimum": 1}}, ["pending_id"]),
    description="Read canonical execution state produced by the deterministic Outcome Reducer.",
    risk_level="none",
    effect_kind="read",
    effect_class="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="control_state",
    audit_event_type="state_observation_read",
)
def state_observation_get(args: dict[str, Any]) -> dict[str, Any]:
    pending = pending_actions.get_pending_action(int(args["pending_id"]))
    if not pending:
        return _result("state_observation_get", status="not_found", state_observation=None)
    execution_key = pending.get("execution_idempotency_key")
    receipt = (
        outcomes.get_effect_receipt_by_execution_key(str(execution_key))
        if execution_key
        else None
    )
    observation = (
        outcomes.get_state_observation(str(receipt["receipt_id"]))
        if receipt
        else None
    )
    return _result(
        "state_observation_get",
        status="success" if observation else "not_found",
        state_observation=observation,
        pending_status=pending.get("status"),
        execution_status=pending.get("execution_status"),
    )
