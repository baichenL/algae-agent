from __future__ import annotations

from typing import Any

from app.core.db import scientific as scientific_db
from app.core.effects import stable_hash
from app.services.agent_runtime.contracts_v2 import ProposalReadiness
from app.services.agent_runtime.safety_v2 import POLICY_VERSION
from app.services.scientific.adapters import get_adapter
from app.services.scientific.models import ExperimentDesignSpec


VALIDATOR_VERSION = "scientific-proposal-validator-v2"
SIMULATOR_VERSION = "scientific-preproposal-simulator-v2"


def _latest(run: dict[str, Any], artifact_type: str) -> dict[str, Any] | None:
    matches = [
        item
        for item in run.get("artifacts") or []
        if item.get("artifact_type") == artifact_type
    ]
    return matches[-1] if matches else None


def _evidence_refs(design: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in design.get("evidence_citations") or []:
        source_id = item.get("source_id") or item.get("document_id") or item.get("title")
        chunk_id = item.get("chunk_id")
        if not source_id:
            continue
        refs.append(
            {
                "ref_id": f"rag:{source_id}:{chunk_id or 'document'}",
                "authority": "rag_advisory",
                "source_id": source_id,
                "chunk_id": chunk_id,
            }
        )
    return refs


def evaluate_scientific_proposal(
    *,
    run: dict[str, Any] | None,
    design: ExperimentDesignSpec,
) -> tuple[ProposalReadiness, dict[str, Any]]:
    missing: list[str] = []
    conflicts: list[str] = []
    investigations: list[str] = []
    details: dict[str, Any] = {}
    if not run:
        missing.append("scientific_run")
        return ProposalReadiness(False, tuple(missing)), details
    if design.scientific_run_id != run.get("id"):
        conflicts.append("design_target_does_not_match_scientific_run")
    dataset = scientific_db.get_dataset(str(run.get("dataset_id") or ""))
    if not dataset:
        missing.append("current_authoritative_dataset")
    else:
        details["dataset"] = dataset
    validation_artifact = _latest(run, "observation")
    validation_observations = [
        item
        for item in run.get("artifacts") or []
        if item.get("artifact_type") == "observation"
        and (item.get("payload") or {}).get("tool_name") == "scientific_design_validate"
    ]
    validation = (
        (validation_observations[-1].get("payload") or {}).get("output") or {}
        if validation_observations
        else {}
    )
    if not validation:
        missing.append("deterministic_design_validation")
    elif not validation.get("valid"):
        conflicts.append("design_validation_failed")
        investigations.append("Apply a PlanPatch using the validator issue fields, then validate again.")
    details["validation"] = validation

    verification_artifact = _latest(run, "verification_report")
    verification = (verification_artifact or {}).get("payload") or {}
    if verification.get("verdict") != "pass":
        missing.append("passing_verification_report")
    unsupported = list(verification.get("unsupported_claims") or [])
    if unsupported:
        conflicts.extend(f"unsupported_claim:{item}" for item in unsupported)

    evidence_refs = _evidence_refs(design.to_dict())
    if not evidence_refs:
        missing.append("valid_evidence_refs")
        investigations.append("Retrieve an applicable SOP or paper and retain stable citation identifiers.")
    details["evidence_refs"] = evidence_refs

    diagnosis_artifact = _latest(run, "diagnosis_report")
    diagnosis = (diagnosis_artifact or {}).get("payload") or {}
    hypothesis_refs = [
        str(item.get("hypothesis_id"))
        for item in diagnosis.get("hypotheses") or []
        if item.get("hypothesis_id")
    ]
    if not hypothesis_refs:
        missing.append("hypothesis_refs")
    details["hypothesis_refs"] = hypothesis_refs

    simulation: dict[str, Any] = {}
    if dataset and validation.get("valid"):
        adapter = get_adapter(str(run.get("adapter_id") or "sim-algae-lab-v1"))
        simulation = adapter.simulate_design(design, dataset=dataset)
        details["adapter"] = adapter.descriptor()
        details["adapter_snapshot"] = adapter.snapshot()
        details["simulation"] = simulation
        if simulation.get("status") != "success":
            conflicts.append("preproposal_simulation_failed")
            investigations.append(
                "Use the simulation validation issues and repairable fields to create a PlanPatch."
            )
    else:
        missing.append("successful_preproposal_simulation")
    if simulation and simulation.get("status") != "success":
        missing.append("successful_preproposal_simulation")
    return (
        ProposalReadiness(
            ready=not missing and not conflicts,
            missing_requirements=tuple(dict.fromkeys(missing)),
            conflicting_evidence=tuple(dict.fromkeys(conflicts)),
            suggested_next_investigations=tuple(dict.fromkeys(investigations)),
        ),
        details,
    )


def build_proposal_envelope(
    *,
    run: dict[str, Any],
    design: ExperimentDesignSpec,
    details: dict[str, Any],
    expires_at: str,
    agent_run_id: str | None,
    tool_call_id: str | None,
) -> dict[str, Any]:
    dataset = details["dataset"]
    validation = details["validation"]
    simulation = details["simulation"]
    simulation_summary = {
        "status": simulation.get("status"),
        "simulation_only": bool(simulation.get("simulation_only", True)),
        "provenance": simulation.get("provenance"),
        "measurement_count": len(simulation.get("measurements") or []),
        "validation": simulation.get("validation") or validation,
        "simulation_hash": stable_hash(simulation),
    }
    return {
        "proposal_version": 2,
        "target": {
            "type": "scientific_run",
            "id": run["id"],
            "version": design.design_hash,
            "strain_id": (run.get("goal") or {}).get("target_strain_id"),
        },
        "design_or_protocol": design.to_dict(),
        "input_resource_versions": {
            "design_hash": design.design_hash,
            "dataset_id": dataset.get("id"),
            "dataset_content_hash": dataset.get("content_hash"),
            "dataset_version": (run.get("goal") or {}).get("dataset_version"),
        },
        "evidence_refs": details["evidence_refs"],
        "hypothesis_refs": details["hypothesis_refs"],
        "validation_result": validation,
        "simulation_result": simulation_summary,
        "policy_version": POLICY_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "risk_summary": {
            "execution_scope": "simulation_or_manual_handoff_only",
            "real_hardware_available": False,
            "formal_email_send": False,
        },
        "uncertainties": [
            "Literature evidence is advisory and does not establish causality.",
            "Simulation output is not a confirmed experimental result.",
        ],
        "expected_resource_versions": {
            "design_hash": design.design_hash,
            "dataset_content_hash": dataset.get("content_hash"),
        },
        "expires_at": expires_at,
        "created_by_agent_run_id": agent_run_id,
        "created_by_tool_call_id": tool_call_id,
    }
