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
    schema=_schema(
        [],
        {
            "strain_id": {"type": ["string", "null"]},
            "ready_only": {"type": "boolean"},
        },
    ),
    risk_level="none",
    effect_kind="read",
    side_effect="none",
    allowed_callers=["chat_runtime", "scientific_runtime", "mcp_local", "frontend_form"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    audit_event_type="scientific_datasets_listed",
)
def handle_scientific_datasets_list(args: dict[str, Any]) -> dict[str, Any]:
    datasets = scientific_db.list_datasets()
    if args.get("strain_id"):
        datasets = [
            item for item in datasets
            if item.get("strain_id") == args["strain_id"]
        ]
    if args.get("ready_only", True):
        datasets = [item for item in datasets if item.get("status") == "ready"]
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
            "target_strain_id": {"type": ["string", "null"]},
            "target_metric": {"type": ["string", "null"]},
            "mode": {"type": "string", "enum": ["diagnose", "diagnose_and_optimize"]},
            "offline_replay": {"type": "boolean"},
            "session_id": {"type": ["string", "null"]},
            "agent_run_id": {"type": ["string", "null"]},
        },
    ),
    risk_level="low",
    effect_kind="compute",
    effect_class="artifact_write",
    side_effect="scientific_artifacts",
    allowed_callers=["chat_runtime", "scientific_runtime", "mcp_local", "frontend_form"],
    exposed_to_llm=True,
    executor_kind="compute",
    audit_event_type="scientific_diagnosis_started",
)
def handle_growth_diagnosis_start(args: dict[str, Any]) -> dict[str, Any]:
    # Keep the scientific service import lazy: the service emits Agent Runtime
    # events, while the runtime imports the governed tool registry at startup.
    from app.services.scientific.service import run_scientific_task

    result = run_scientific_task(
        dataset_id=args["dataset_id"],
        target_strain_id=args.get("target_strain_id"),
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
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
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
    exposed_to_llm=True,
    executor_kind="proposal",
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
        proposal_version=2,
        created_by_tool_call_id=args.get("created_by_tool_call_id"),
    )
    if proposal.get("status") == "proposal_not_ready":
        payload = {
            **proposal,
            "action": "experiment_proposal_request",
            "message": "The proposal readiness gate rejected this candidate; no pending was created.",
        }
        return {**payload, "response_payload": payload}
    payload = {
        **proposal, "action": "scientific_experiment_plan", "require_confirmation": True,
        "message": "A frozen simulation-only experiment plan is waiting for approver review.",
    }
    return {**payload, "response_payload": payload}


@register_agent_tool(
    name="candidate_design_simulate",
    schema=_schema(
        ["scientific_run_id", "design"],
        {
            "scientific_run_id": RUN_ID,
            "candidate_id": {"type": "string", "minLength": 1},
            "design": {"type": "object"},
        },
    ),
    description="Validate and simulate one candidate design; returns repairable constraint failures.",
    risk_level="low",
    effect_kind="compute",
    effect_class="artifact_write",
    side_effect="scientific_artifacts",
    allowed_callers=["chat_runtime", "scientific_runtime"],
    exposed_to_llm=True,
    executor_kind="compute",
    audit_event_type="candidate_simulated",
)
def handle_candidate_design_simulate(args: dict[str, Any]) -> dict[str, Any]:
    from app.services.scientific.adapters import get_adapter
    from app.services.scientific.service import _design_from_payload
    from app.services.scientific.models import experiment_design_hash

    run = scientific_db.get_scientific_run(args["scientific_run_id"])
    if not run:
        payload = {
            "status": "not_found",
            "action": "candidate_design_simulate",
            "error_code": "scientific_run_not_found",
        }
        return {**payload, "response_payload": payload}
    dataset = scientific_db.get_dataset(str(run.get("dataset_id") or ""))
    if not dataset:
        payload = {
            "status": "not_found",
            "action": "candidate_design_simulate",
            "error_code": "scientific_dataset_not_found",
        }
        return {**payload, "response_payload": payload}
    try:
        design_payload = dict(args["design"])
        if design_payload.get("scientific_run_id") != args["scientific_run_id"]:
            raise ValueError("design_target_mismatch")
        actual_hash = experiment_design_hash(design_payload)
        if design_payload.get("design_hash") and design_payload["design_hash"] != actual_hash:
            raise ValueError("design_hash_mismatch")
        design_payload["design_hash"] = actual_hash
        design = _design_from_payload(design_payload)
    except Exception as exc:
        payload = {
            "status": "error",
            "action": "candidate_design_simulate",
            "error_code": "invalid_design",
            "error_details": {"message": str(exc)},
        }
        return {**payload, "response_payload": payload}
    adapter = get_adapter(str(run.get("adapter_id") or "sim-algae-lab-v1"))
    validation = adapter.validate_design(design)
    simulation = (
        adapter.simulate_design(design, dataset=dataset)
        if validation.get("valid")
        else {
            "status": "failed",
            "validation": validation,
            "simulation_only": True,
        }
    )
    artifact = scientific_db.add_artifact(
        args["scientific_run_id"],
        "candidate_simulation",
        {
            "candidate_id": args.get("candidate_id"),
            "design_hash": design.design_hash,
            "validation": validation,
            "simulation": simulation,
        },
    )
    payload = {
        "status": "success" if simulation.get("status") == "success" else "simulation_failed",
        "action": "candidate_design_simulate",
        "candidate_id": args.get("candidate_id"),
        "design_hash": design.design_hash,
        "validation": validation,
        "simulation": simulation,
        "artifact_ref": artifact["content_hash"],
        "repairable_fields": [
            item.get("code") for item in validation.get("issues") or []
        ],
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
