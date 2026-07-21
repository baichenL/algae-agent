from __future__ import annotations

from typing import Any

from app.core.db import scientific as scientific_db
from app.tools.registry_core import register_agent_tool


def _schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


DATASET_ID = {"type": "string", "minLength": 1}
RUN_ID = {"type": "string", "minLength": 1}


@register_agent_tool(
    name="scientific_datasets_list",
    schema=_schema([], {}),
    risk_level="none",
    effect_kind="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "scientific_runtime", "mcp_local", "frontend_form"],
    audit_event_type="scientific_datasets_listed",
)
def handle_scientific_datasets_list(_: dict[str, Any]) -> dict[str, Any]:
    datasets = scientific_db.list_datasets()
    return {
        "status": "success", "action": "scientific_datasets_list", "datasets": datasets,
        "response_payload": {"status": "success", "action": "scientific_datasets_list", "datasets": datasets},
    }


@register_agent_tool(
    name="growth_diagnosis_start",
    schema=_schema(
        ["dataset_id"],
        {
            "dataset_id": DATASET_ID,
            "target_metric": {"type": ["string", "null"]},
            "mode": {"type": "string", "enum": ["diagnose", "diagnose_and_optimize"]},
            "offline_replay": {"type": "boolean"},
            "session_id": {"type": ["string", "null"]},
            "agent_run_id": {"type": ["string", "null"]},
        },
    ),
    risk_level="low",
    effect_kind="read",
    side_effect="scientific_artifacts",
    allowed_callers=["chat_runtime", "scientific_runtime", "mcp_local", "frontend_form"],
    audit_event_type="scientific_diagnosis_started",
)
def handle_growth_diagnosis_start(args: dict[str, Any]) -> dict[str, Any]:
    # Keep the scientific service import lazy: the service emits Agent Runtime
    # events, while the runtime imports the governed tool registry at startup.
    from app.services.scientific.service import run_scientific_task

    result = run_scientific_task(
        dataset_id=args["dataset_id"],
        target_metric=args.get("target_metric"),
        mode=args.get("mode") or "diagnose",
        offline_replay=bool(args.get("offline_replay")),
        session_id=args.get("session_id") or "scientific-tool",
        agent_run_id=args.get("agent_run_id"),
        create_pending=False,
    )
    payload = {"status": "success", "action": "growth_diagnosis", "scientific_run": result}
    return {**payload, "response_payload": payload}


@register_agent_tool(
    name="experiment_design_preview",
    schema=_schema(["scientific_run_id"], {"scientific_run_id": RUN_ID}),
    risk_level="none",
    effect_kind="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "scientific_runtime", "mcp_local", "frontend_form"],
    audit_event_type="scientific_design_previewed",
)
def handle_experiment_design_preview(args: dict[str, Any]) -> dict[str, Any]:
    run = scientific_db.get_scientific_run(args["scientific_run_id"])
    designs = [item for item in (run or {}).get("artifacts") or [] if item.get("artifact_type") == "experiment_design"]
    design = designs[-1].get("payload") if designs else None
    status = "success" if design else "not_found"
    payload = {"status": status, "action": "experiment_design_preview", "design": design}
    return {**payload, "response_payload": payload}


@register_agent_tool(
    name="experiment_proposal_request",
    schema=_schema(["scientific_run_id"], {"scientific_run_id": RUN_ID, "agent_run_id": {"type": ["string", "null"]}}),
    risk_level="medium",
    effect_kind="propose",
    side_effect="db_pending",
    requires_approval=True,
    allowed_callers=["chat_runtime", "scientific_runtime", "mcp_local", "frontend_form"],
    idempotency_fields=["scientific_run_id"],
    audit_event_type="scientific_experiment_proposed",
)
def handle_experiment_proposal_request(args: dict[str, Any]) -> dict[str, Any]:
    from app.services.scientific.service import create_experiment_proposal

    run = scientific_db.get_scientific_run(args["scientific_run_id"])
    designs = [item for item in (run or {}).get("artifacts") or [] if item.get("artifact_type") == "experiment_design"]
    if not designs:
        return {
            "status": "error", "action": "experiment_proposal_request", "message": "No verified experiment design is available.",
            "response_payload": {"status": "error", "action": "experiment_proposal_request"},
        }
    proposal = create_experiment_proposal(
        args["scientific_run_id"], designs[-1]["payload"],
        requester="tool_gateway", source="scientific_tool", agent_run_id=args.get("agent_run_id"),
    )
    payload = {
        **proposal, "action": "scientific_experiment_plan", "require_confirmation": True,
        "message": "A frozen simulation-only experiment plan is waiting for approver review.",
    }
    return {**payload, "response_payload": payload}


# Internal scientific operations are registered with explicit capability metadata.
# The orchestrator owns their typed inputs and persists their outputs as artifacts.
for _name, _skill in (
    ("scientific_data_quality", "growth_anomaly_diagnosis"),
    ("scientific_growth_metrics", "growth_anomaly_diagnosis"),
    ("scientific_anomaly_detection", "growth_anomaly_diagnosis"),
    ("scientific_evidence_search", "evidence_grounded_hypothesis"),
    ("scientific_hypothesis_ranking", "evidence_grounded_hypothesis"),
    ("scientific_experiment_optimize", "constrained_experiment_optimization"),
    ("scientific_design_validate", "constrained_experiment_optimization"),
    ("scientific_design_repair", "simulation_failure_repair"),
):
    if _name not in __import__("app.tools.registry_core", fromlist=["AGENT_TOOL_REGISTRY"]).AGENT_TOOL_REGISTRY:
        register_agent_tool(
            name=_name,
            schema={"type": "object", "additionalProperties": True},
            risk_level="low",
            effect_kind="read",
            side_effect="scientific_artifacts",
            allowed_callers=["scientific_runtime"],
            audit_event_type=_name,
        )(lambda args, operation=_name, skill=_skill: {
            "status": "success", "action": operation, "skill": skill, "output": args,
            "response_payload": {"status": "success", "action": operation, "skill": skill, "output": args},
        })
