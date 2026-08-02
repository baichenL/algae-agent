from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from typing import Any

from app.core.db import pending_actions
from app.core.db import scientific as scientific_db
from app.models.rag_schema import RagQueryRequest
from app.services.agent_runtime.events import finish_run, record_run_event, start_run
from app.services.rag.service import answer_rag_question, citation_from_retrieved_chunk
from app.core.database import search_rag_chunks
from app.services.scientific.adapters import get_adapter
from app.services.scientific.analysis import (
    assess_data_quality,
    build_diagnosis,
    compute_growth_metrics,
    detect_anomalies,
)
from app.services.scientific.models import (
    ExperimentDesignSpec,
    GoalContract,
    PlanPatch,
    ScientificObservation,
    ScientificPlanNode,
    VerificationCheck,
    VerificationReport,
    build_default_plan,
    canonical_hash,
    experiment_design_hash,
)
from app.services.scientific.optimizer import fit_response_surface, propose_conditions
from app.services.scientific.proposal_gate import (
    build_proposal_envelope,
    evaluate_scientific_proposal,
)
from app.core.effects import stable_hash
from app.core.time_utils import local_now, local_time_string
import datetime


def _event(run: dict[str, Any], event_type: str, payload: dict[str, Any]) -> None:
    record_run_event(
        run.get("agent_run_id"),
        session_id=run.get("session_id"),
        event_type=event_type,
        layer="scientific_agent",
        payload={"scientific_run_id": run["id"], **payload},
    )


def _artifact(run: dict[str, Any], artifact_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = scientific_db.add_artifact(run["id"], artifact_type, payload)
    _event(
        run,
        f"scientific_{artifact_type}_created",
        {"artifact_type": artifact_type, "version": result["version"], "content_hash": result["content_hash"]},
    )
    return result


def _observation(
    run: dict[str, Any], step_id: str, tool_name: str, output: dict[str, Any],
    *, quality_flags: list[str] | None = None, evidence_refs: list[dict[str, Any]] | None = None,
    provenance: str = "deterministic_analysis",
) -> dict[str, Any]:
    observation = ScientificObservation(
        step_id=step_id, tool_name=tool_name, status="success", output=output,
        quality_flags=tuple(quality_flags or []), evidence_refs=tuple(evidence_refs or []), provenance=provenance,
    ).to_dict()
    _artifact(run, "observation", observation)
    return observation


def _replay_split(dataset: dict[str, Any], run_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    batches = list(dataset.get("batches") or [])
    ordered = sorted(
        batches,
        key=lambda item: hashlib.sha256(f"{run_seed}:{item['id']}".encode("utf-8")).hexdigest(),
    )
    visible_count = min(17, max(1, len(ordered) - min(8, max(0, len(ordered) // 3))))
    visible = ordered[:visible_count]
    hidden = ordered[visible_count:]
    full = deepcopy(dataset)
    full["replay_split"] = {
        "visible_batch_ids": [item["id"] for item in visible],
        "hidden_batch_ids": [item["id"] for item in hidden],
    }
    analysis_dataset = deepcopy(dataset)
    analysis_dataset["batches"] = visible
    analysis_dataset["replay_split"] = full["replay_split"]
    return analysis_dataset, full


def _retrieve_evidence(
    factor_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not factor_names:
        return [], {
            "source": "rag",
            "status": "empty",
            "used": False,
            "impact": "No modeled factors were available for evidence retrieval.",
        }
    # Start with a language-neutral broad query, then query individual factor
    # tokens. A combined multilingual query can otherwise be rejected by the
    # evidence-sufficiency checker because one untranslated token is missing.
    questions = ["microalgae algae growth evidence association controlled experiment"]
    questions.extend(f"microalgae algae growth {factor}" for factor in factor_names)
    citations: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    failures = 0
    completed_queries = 0
    offline_fts = os.getenv("SCIENTIFIC_EVIDENCE_OFFLINE_FTS", "false").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not offline_fts:
        for question in questions:
            try:
                response = answer_rag_question(RagQueryRequest(question=question, top_k=5, doc_types=["paper", "manual"]))
                completed_queries += 1
            except Exception:
                failures += 1
                continue
            if response.blocked:
                failures += 1
                continue
            for item in response.citations:
                payload = item.model_dump()
                key = (payload.get("source_id"), payload.get("chunk_id"))
                if key not in seen:
                    seen.add(key)
                    citations.append(payload)
                if len(citations) >= 5:
                    break
            if len(citations) >= 5:
                break
    # The answerability layer can conservatively decline to synthesize an
    # answer even when the index contains useful background evidence. Scientific
    # diagnosis only needs read-only provenance, so fall back to the same real
    # retriever and retain complete source locators instead of fabricating a
    # citation or treating an empty answer as an empty index.
    if not citations:
        for question in questions:
            try:
                chunks = search_rag_chunks(
                    question,
                    top_k=5,
                    doc_types=["paper", "manual"],
                )
            except Exception:
                failures += 1
                continue
            completed_queries += 1
            for chunk in chunks:
                source_id = int(chunk.get("knowledge_source_id") or chunk.get("document_id") or 0)
                if not source_id:
                    continue
                payload = citation_from_retrieved_chunk(source_id, chunk).model_dump()
                key = (payload.get("source_id"), payload.get("chunk_id"))
                if key not in seen:
                    seen.add(key)
                    citations.append(payload)
                if len(citations) >= 5:
                    break
            if citations:
                break
    if citations and failures:
        status = "degraded"
    elif citations:
        status = "available"
    elif failures:
        status = "failed"
    else:
        status = "empty"
    return citations, {
        "source": "rag",
        "status": status,
        "used": bool(citations),
        "query_count": len(questions),
        "completed_queries": completed_queries,
        "failed_queries": failures,
        "impact": (
            "Literature support is incomplete; conclusions remain data associations."
            if status in {"failed", "degraded", "empty"}
            else "Literature evidence contributed to hypothesis support."
        ),
    }


def _verification(
    *, quality: dict[str, Any], model: dict[str, Any], diagnosis: dict[str, Any],
    evidence: list[dict[str, Any]], design_validation: dict[str, Any] | None = None,
) -> VerificationReport:
    checks = [
        VerificationCheck("data_quality", bool(quality.get("batch_count")) and quality.get("passed", False), "Critical quality checks passed." if quality.get("passed") else "Data quality requires caution."),
        VerificationCheck("statistical_support", model.get("status") == "ok", model.get("mode") or model.get("reason", "model unavailable")),
        VerificationCheck("causal_language_guard", not diagnosis.get("causal_claim", False), "Diagnosis is expressed as candidate causes, not proven causality."),
        VerificationCheck("evidence_boundary", True, "Missing literature lowers support but never authorizes an action."),
    ]
    if design_validation is not None:
        checks.append(
            VerificationCheck("design_feasibility", bool(design_validation.get("valid")), "Design fits the active capability profile." if design_validation.get("valid") else "Design requires a plan patch.")
        )
    critical_failure = any(item.name in {"statistical_support", "causal_language_guard"} and not item.passed for item in checks)
    revision = any(item.name == "design_feasibility" and not item.passed for item in checks)
    verdict = "block" if critical_failure else "revise" if revision else "pass"
    return VerificationReport(
        verdict=verdict,
        checks=tuple(checks),
        missing_information=() if evidence else ("No directly citable SOP/paper evidence was retrieved; hypotheses remain data-only associations.",),
        unsupported_claims=(),
    )


def _build_design(
    run_id: str,
    goal: GoalContract,
    optimization: dict[str, Any],
    evidence: list[dict[str, Any]],
    *, cycle_index: int,
) -> ExperimentDesignSpec:
    sampling_hours = sorted(
        {
            float(item.get("elapsed_hours"))
            for item in optimization.get("source_measurements", [])
            if item.get("elapsed_hours") is not None
        }
    ) or [0.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0]
    return ExperimentDesignSpec(
        design_id=f"design_{run_id}_{cycle_index}",
        scientific_run_id=run_id,
        target_metric=goal.target_metric,
        direction=goal.direction,
        conditions=list(optimization["conditions"]),
        replicates=3,
        sampling_hours=sampling_hours,
        required_capabilities=["prepare_conditions", "inoculate", "incubate", "measure_growth", "record_results"],
        evidence_citations=evidence,
        simulation_only=True,
    ).freeze()


def _repair_capacity_design(
    plan_version: int,
    design: ExperimentDesignSpec,
    validation: dict[str, Any],
) -> tuple[ExperimentDesignSpec, PlanPatch]:
    available = int(validation.get("available_culture_vessels") or 0)
    max_conditions = max(1, available // max(1, design.replicates))
    controls = [item for item in design.conditions if item.role == "control"]
    candidates = sorted(
        [item for item in design.conditions if item.role == "candidate"],
        key=lambda item: item.acquisition_score,
        reverse=True,
    )
    keep_candidate_count = max(0, max_conditions - len(controls))
    removed = candidates[keep_candidate_count:]
    repaired = ExperimentDesignSpec(
        design_id=f"{design.design_id}_repaired",
        scientific_run_id=design.scientific_run_id,
        target_metric=design.target_metric,
        direction=design.direction,
        conditions=candidates[:keep_candidate_count] + controls[:1],
        replicates=design.replicates,
        sampling_hours=design.sampling_hours,
        required_capabilities=design.required_capabilities,
        evidence_citations=design.evidence_citations,
        simulation_only=True,
    ).freeze()
    patch = PlanPatch(
        base_version=plan_version,
        new_version=plan_version + 1,
        reason="CAPACITY_EXCEEDED: stage the screening design within the active device profile.",
        replace_nodes=(
            ScientificPlanNode(
                "repair_capacity",
                "simulation_failure_repair",
                "scientific_design_repair",
                "Reduce candidate count while preserving one control and three replicates.",
                ("validate",),
                "completed",
            ),
        ),
        metadata={
            "removed_condition_ids": [item.condition_id for item in removed],
            "retained_condition_ids": [item.condition_id for item in repaired.conditions],
            "control_condition_ids": [item.condition_id for item in controls[:1]],
            "before_condition_count": len(design.conditions),
            "after_condition_count": len(repaired.conditions),
            "replicates": design.replicates,
            "before_required_vessels": len(design.conditions) * design.replicates,
            "after_required_vessels": len(repaired.conditions) * repaired.replicates,
            "before_design_hash": design.design_hash,
            "after_design_hash": repaired.design_hash,
            "available_culture_vessels": available,
        },
    )
    return repaired, patch


def create_experiment_proposal(
    scientific_run_id: str,
    design: ExperimentDesignSpec | dict[str, Any],
    *,
    requester: str = "scientific_agent",
    source: str = "scientific_runtime",
    agent_run_id: str | None = None,
    replay_split: dict[str, Any] | None = None,
    proposal_version: int = 1,
    created_by_tool_call_id: str | None = None,
) -> dict[str, Any]:
    design_payload = design.to_dict() if isinstance(design, ExperimentDesignSpec) else dict(design)
    claimed_hash = design_payload.get("design_hash")
    unhashed = {key: value for key, value in design_payload.items() if key != "design_hash"}
    actual_hash = experiment_design_hash(unhashed)
    if claimed_hash != actual_hash:
        raise ValueError("design_hash_mismatch")
    proposal_envelope: dict[str, Any] | None = None
    proposal_hash: str | None = None
    expected_versions: dict[str, Any] | None = None
    expires_at = local_time_string(
        local_now()
        + datetime.timedelta(
            seconds=max(60, int(os.getenv("PENDING_DEFAULT_TTL_SECONDS", "86400")))
        )
    )
    if int(proposal_version) >= 2:
        typed_design = _design_from_payload(design_payload)
        run = scientific_db.get_scientific_run(scientific_run_id)
        readiness, readiness_details = evaluate_scientific_proposal(
            run=run,
            design=typed_design,
        )
        if not readiness.ready:
            return {
                "status": "proposal_not_ready",
                "scientific_run_id": scientific_run_id,
                "missing_requirements": list(readiness.missing_requirements),
                "conflicting_evidence": list(readiness.conflicting_evidence),
                "suggested_next_investigations": list(
                    readiness.suggested_next_investigations
                ),
            }
        _artifact(
            run,
            "preproposal_simulation_report",
            readiness_details["simulation"],
        )
        proposal_envelope = build_proposal_envelope(
            run=run,
            design=typed_design,
            details=readiness_details,
            expires_at=expires_at,
            agent_run_id=agent_run_id,
            tool_call_id=created_by_tool_call_id,
        )
        proposal_hash = stable_hash(proposal_envelope)
        expected_versions = dict(
            proposal_envelope.get("expected_resource_versions") or {}
        )
    payload = {
        "type": "scientific_experiment_plan",
        "data": {
            "scientific_run_id": scientific_run_id,
            "design": design_payload,
            "design_hash": actual_hash,
            "adapter_id": "offline-replay-v1" if replay_split and replay_split.get("hidden_batch_ids") else "sim-algae-lab-v1",
            "simulation_only": True,
            "replay_split": replay_split or {},
            "proposal_hash": proposal_hash,
        },
    }
    pending_id = pending_actions.insert_pending_action(
        "scientific_experiment_plan",
        payload,
        requester=requester,
        risk_level="medium",
        source=source,
        agent_run_id=agent_run_id,
        execution_idempotency_key=f"scientific:{scientific_run_id}:{actual_hash}",
        domain_dedupe_key=f"scientific-design:{scientific_run_id}:{actual_hash}",
        expires_at=expires_at,
        proposal_version=int(proposal_version),
        proposal_hash=proposal_hash,
        proposal_envelope=proposal_envelope,
        policy_version=(proposal_envelope or {}).get("policy_version"),
        expected_resource_versions=expected_versions,
        created_by_tool_call_id=created_by_tool_call_id,
    )
    scientific_db.update_scientific_run(scientific_run_id, status="waiting_approval")
    return {
        "status": "waiting_approval",
        "pending_id": pending_id,
        "scientific_run_id": scientific_run_id,
        "design_hash": actual_hash,
        "simulation_only": True,
        "proposal_version": int(proposal_version),
        "proposal_hash": proposal_hash,
        "expires_at": expires_at,
    }


def _prepare_next_design(
    *, run: dict[str, Any], dataset: dict[str, Any], goal: GoalContract,
    metrics: list[dict[str, Any]], evidence: list[dict[str, Any]], excluded_conditions: list[dict[str, float]],
    candidate_pool: list[dict[str, float]] | None = None,
) -> tuple[ExperimentDesignSpec | None, dict[str, Any], PlanPatch | None]:
    model = fit_response_surface(metrics)
    optimization = propose_conditions(
        model,
        direction=goal.direction,
        run_seed=f"{goal.run_seed}:{run.get('cycle_index', 0)}",
        excluded_conditions=excluded_conditions,
        candidate_pool=candidate_pool,
    )
    optimization["source_measurements"] = [
        item for batch in dataset.get("batches") or [] for item in batch.get("measurements") or []
    ]
    if optimization.get("status") != "ok":
        return None, {"model": model, "optimization": optimization}, None
    design = _build_design(run["id"], goal, optimization, evidence, cycle_index=int(run.get("cycle_index") or 0))
    adapter = get_adapter(run.get("adapter_id") or "sim-algae-lab-v1")
    validation = adapter.validate_design(design)
    patch = None
    if not validation["valid"] and any(item.get("code") == "CAPACITY_EXCEEDED" for item in validation["issues"]):
        design, patch = _repair_capacity_design(1 + int(run.get("cycle_index") or 0), design, validation)
        validation = adapter.validate_design(design)
    return design, {"model": model, "optimization": optimization, "validation": validation}, patch


def run_scientific_task(
    *,
    dataset_id: str,
    target_strain_id: str | None = None,
    mode: str = "diagnose_and_optimize",
    target_metric: str | None = None,
    direction: str = "maximize",
    target_batch_ids: list[str] | None = None,
    max_cycles: int = 2,
    run_seed: int = 2025,
    session_id: str = "scientific-api",
    agent_run_id: str | None = None,
    offline_replay: bool = False,
    create_pending: bool = True,
) -> dict[str, Any]:
    dataset = scientific_db.get_dataset(dataset_id)
    if not dataset:
        raise ValueError("scientific_dataset_not_found")
    dataset_strain_id = str(dataset.get("strain_id") or "")
    target_strain_id = str(target_strain_id or dataset_strain_id or "") or None
    if target_strain_id and dataset_strain_id != target_strain_id:
        raise ValueError("scientific_target_dataset_mismatch")
    metric = target_metric or str((dataset.get("mapping") or {}).get("metric_name") or "biomass")
    run_id = f"sci_{uuid.uuid4().hex[:16]}"
    trace_run_id = agent_run_id or start_run(session_id, f"scientific:{mode}:{dataset_id}")
    goal = GoalContract(
        dataset_id=dataset_id,
        target_strain_id=target_strain_id,
        dataset_version=str(dataset.get("created_at") or ""),
        dataset_content_hash=str(dataset.get("content_hash") or ""),
        target_metric=metric,
        direction="minimize" if direction == "minimize" else "maximize",
        target_batch_ids=tuple(target_batch_ids or []),
        constraints={"max_culture_vessels": 12, "simulation_only": True},
        max_cycles=max(1, min(int(max_cycles), 2)),
        run_seed=int(run_seed),
    )
    adapter_id = "offline-replay-v1" if offline_replay else "sim-algae-lab-v1"
    run = {
        "id": run_id, "agent_run_id": trace_run_id, "session_id": session_id, "dataset_id": dataset_id,
        "status": "running", "mode": mode, "cycle_index": 0, "goal": goal.to_dict(), "adapter_id": adapter_id,
    }
    scientific_db.insert_scientific_run(run)
    _artifact(run, "goal_contract", goal.to_dict())
    plan = build_default_plan()
    _artifact(run, "plan_graph", plan.to_dict())

    analysis_dataset, full_dataset = _replay_split(dataset, goal.run_seed) if offline_replay else (dataset, dataset)
    if offline_replay:
        _artifact(run, "replay_split", full_dataset["replay_split"])
    quality = assess_data_quality(analysis_dataset, metric)
    _observation(run, "quality", "scientific_data_quality", quality, quality_flags=[item["code"] for item in quality["flags"]])
    metrics = compute_growth_metrics(analysis_dataset, metric)
    _observation(run, "metrics", "scientific_growth_metrics", {"metrics": metrics})
    model = fit_response_surface(metrics)
    anomalies = detect_anomalies(metrics, model.get("residuals"))
    _observation(run, "anomaly", "scientific_anomaly_detection", {"anomalies": anomalies, "model_diagnostics": model})
    evidence, rag_status = _retrieve_evidence(model.get("factor_names") or [])
    source_statuses = [
        {
            "source": "scientific_dataset",
            "status": "available",
            "used": True,
            "dataset_id": dataset_id,
            "strain_id": dataset_strain_id,
            "content_hash": dataset.get("content_hash"),
        },
        rag_status,
    ]
    _artifact(run, "source_status", {"sources": source_statuses})
    _observation(run, "evidence", "scientific_evidence_search", {"citations": evidence}, evidence_refs=evidence, provenance="rag_read_only")
    diagnosis = build_diagnosis(
        anomalies=anomalies, quality=quality,
        factor_importance=model.get("factor_importance") or [], evidence=evidence,
    ).to_dict()
    def enrich_result(result: dict[str, Any]) -> dict[str, Any]:
        result.update({
            "direct_answer": diagnosis.get("conclusion"),
            "diagnosis": diagnosis,
            "source_statuses": source_statuses,
            "target_strain_id": target_strain_id,
        })
        return result
    _artifact(run, "diagnosis_report", diagnosis)
    _artifact(
        run,
        "hypothesis_ledger",
        {
            "target_strain_id": target_strain_id,
            "hypotheses": [
                {
                    **item,
                    "status": "active",
                    "supporting_evidence": item.get("evidence_refs") or [],
                    "counterevidence": [],
                    "confidence_history": [item.get("score")],
                    "next_validation_action": (
                        (item.get("limitations") or ["Run a controlled validation experiment."])[0]
                    ),
                }
                for item in diagnosis.get("hypotheses") or []
            ],
        },
    )
    _observation(run, "diagnosis", "scientific_hypothesis_ranking", diagnosis, evidence_refs=evidence)

    initial_verification = _verification(quality=quality, model=model, diagnosis=diagnosis, evidence=evidence)
    _artifact(run, "verification_report", initial_verification.to_dict())
    if mode == "diagnose" or initial_verification.verdict == "block":
        status = "succeeded" if initial_verification.verdict != "block" else "blocked"
        scientific_db.update_scientific_run(run_id, status=status)
        finish_run(trace_run_id, status=status, final_route="scientific_task", risk_level="low", response_summary=diagnosis["conclusion"])
        result = scientific_db.get_scientific_run(run_id) or {"id": run_id, "status": status}
        return enrich_result(result)

    design, design_context, patch = _prepare_next_design(
        run=run, dataset=analysis_dataset, goal=goal, metrics=metrics, evidence=evidence, excluded_conditions=[],
        candidate_pool=[
            batch.get("condition") or {}
            for batch in full_dataset.get("batches") or []
            if batch.get("id") in set((full_dataset.get("replay_split") or {}).get("hidden_batch_ids") or [])
        ] if offline_replay else None,
    )
    _artifact(run, "optimization_report", {key: value for key, value in design_context.items() if key != "optimization"} | {
        "optimization": {
            "status": design_context.get("optimization", {}).get("status"),
            "candidate_count": len(design_context.get("optimization", {}).get("conditions") or []),
        }
    })
    if design is None:
        scientific_db.update_scientific_run(run_id, status="blocked")
        return enrich_result(
            scientific_db.get_scientific_run(run_id)
            or {"id": run_id, "status": "blocked"}
        )
    if patch:
        _artifact(run, "plan_patch", patch.to_dict())
        plan.version = patch.new_version
        plan.nodes.extend(patch.replace_nodes)
        _artifact(run, "plan_graph", plan.to_dict())
    _artifact(run, "experiment_design", design.to_dict())
    _observation(run, "validate", "scientific_design_validate", design_context["validation"])
    final_verification = _verification(
        quality=quality, model=model, diagnosis=diagnosis, evidence=evidence,
        design_validation=design_context["validation"],
    )
    _artifact(run, "verification_report", final_verification.to_dict())
    if final_verification.verdict != "pass":
        scientific_db.update_scientific_run(run_id, status="blocked")
        return enrich_result(
            scientific_db.get_scientific_run(run_id)
            or {"id": run_id, "status": "blocked"}
        )

    proposal = None
    if create_pending:
        proposal = create_experiment_proposal(
            run_id, design, agent_run_id=trace_run_id,
            replay_split=full_dataset.get("replay_split"),
        )
        _observation(run, "proposal", "scientific_proposal_create", proposal)
    else:
        scientific_db.update_scientific_run(run_id, status="design_ready")
    result = scientific_db.get_scientific_run(run_id) or {"id": run_id}
    result["proposal"] = proposal
    return enrich_result(result)


def _design_from_payload(payload: dict[str, Any]) -> ExperimentDesignSpec:
    from app.services.scientific.models import ExperimentCondition
    return ExperimentDesignSpec(
        design_id=payload["design_id"], scientific_run_id=payload["scientific_run_id"],
        target_metric=payload["target_metric"], direction=payload["direction"],
        conditions=[ExperimentCondition(**item) for item in payload.get("conditions") or []],
        replicates=int(payload["replicates"]), sampling_hours=[float(x) for x in payload.get("sampling_hours") or []],
        required_capabilities=list(payload.get("required_capabilities") or []),
        evidence_citations=list(payload.get("evidence_citations") or []),
        simulation_only=True, design_hash=payload.get("design_hash") or "",
    )


def approve_and_simulate_proposal(
    pending_id: int,
    *,
    reviewed_by: str = "approver",
    idempotency_key: str | None = None,
    finalize_pending: bool = True,
) -> dict[str, Any]:
    pending = pending_actions.get_pending_action(pending_id)
    if not pending:
        return {"status": "error", "reason": "pending_not_found"}
    data = ((pending.get("payload") or {}).get("data") or {})
    if (pending.get("payload") or {}).get("type") != "scientific_experiment_plan":
        return {"status": "error", "reason": "not_scientific_experiment_plan"}
    if pending.get("status") == "denied":
        return {"status": "error", "reason": "pending_denied"}
    if pending.get("executed_at") and pending.get("execution_result"):
        return {**pending["execution_result"], "idempotent": True}
    design_payload = dict(data.get("design") or {})
    claimed_hash = data.get("design_hash")
    actual_hash = experiment_design_hash(design_payload)
    if claimed_hash != actual_hash or design_payload.get("design_hash") != actual_hash:
        return {"status": "error", "reason": "design_hash_mismatch"}
    scientific_run_id = str(data["scientific_run_id"])
    run = scientific_db.get_scientific_run(scientific_run_id)
    if not run:
        return {"status": "error", "reason": "scientific_run_not_found"}
    if pending.get("status") == "pending":
        pending_actions.update_pending_action_status(pending_id, "approved", reviewed_by=reviewed_by)
    design = _design_from_payload(design_payload)
    dataset = scientific_db.get_dataset(run["dataset_id"])
    if not dataset:
        return {"status": "error", "reason": "scientific_dataset_not_found"}
    replay_split = data.get("replay_split") or {}
    if replay_split:
        dataset["replay_split"] = replay_split
    adapter = get_adapter(data.get("adapter_id") or run.get("adapter_id") or "sim-algae-lab-v1")
    simulation = adapter.simulate_design(design, dataset=dataset)
    runtime = {
        "id": scientific_run_id,
        "agent_run_id": run.get("agent_run_id"),
        "session_id": run.get("session_id"),
    }
    if simulation.get("status") != "success":
        _artifact(runtime, "simulation_report", simulation)
        scientific_db.update_scientific_run(scientific_run_id, status="simulation_failed")
        return {"status": "error", "reason": "digital_twin_failed", "simulation": simulation}
    result_id = str(simulation["simulation_id"])
    scientific_db.insert_virtual_result(
        {
            "id": result_id, "scientific_run_id": scientific_run_id, "pending_id": pending_id,
            "design_hash": actual_hash, "adapter_id": adapter.descriptor()["id"],
            "provenance": simulation["provenance"], "measurements": simulation["measurements"],
            "summary": {
                "measurement_count": len(simulation["measurements"]),
                "matched_hidden_batch_ids": simulation.get("matched_hidden_batch_ids") or [],
                "simulation_only": True,
            },
        }
    )
    _artifact(runtime, "simulation_report", simulation)
    _observation(
        runtime, "digital_twin", "scientific_digital_twin_execute", simulation,
        provenance=simulation["provenance"],
    )
    cycle = int(run.get("cycle_index") or 0) + 1
    goal = GoalContract(**{
        **run["goal"],
        "target_batch_ids": tuple(run["goal"].get("target_batch_ids") or []),
    })
    patch = PlanPatch(
        base_version=cycle + 1, new_version=cycle + 2,
        reason="A newly approved digital-twin observation changes the available evidence and requires model refitting.",
        add_nodes=(ScientificPlanNode(
            f"refit_cycle_{cycle}", "constrained_experiment_optimization", "scientific_model_refit",
            "Refit the response model with explicitly simulation-tagged observations.", ("digital_twin",), "completed",
        ),),
        metadata={"simulation_id": result_id, "provenance": simulation["provenance"], "simulation_only": True},
    )
    _artifact(runtime, "plan_patch", patch.to_dict())
    scientific_db.update_scientific_run(scientific_run_id, status="cycle_completed", cycle_index=cycle)

    next_pending = None
    if cycle < goal.max_cycles:
        latest_design_artifacts = [
            item for item in run.get("artifacts") or [] if item.get("artifact_type") == "experiment_design"
        ]
        excluded = [
            condition.get("factors") or {}
            for item in latest_design_artifacts
            for condition in (item.get("payload") or {}).get("conditions") or []
            if condition.get("role") == "candidate"
        ]
        analysis_dataset = dataset
        if replay_split:
            visible_ids = set(replay_split.get("visible_batch_ids") or []) | set(simulation.get("matched_hidden_batch_ids") or [])
            analysis_dataset = deepcopy(dataset)
            analysis_dataset["batches"] = [batch for batch in dataset.get("batches") or [] if batch["id"] in visible_ids]
        metrics = compute_growth_metrics(analysis_dataset, goal.target_metric)
        evidence_artifacts = [item for item in run.get("artifacts") or [] if item.get("artifact_type") == "observation" and (item.get("payload") or {}).get("tool_name") == "scientific_evidence_search"]
        evidence = ((evidence_artifacts[-1].get("payload") or {}).get("output") or {}).get("citations") if evidence_artifacts else []
        run_context = {**run, "cycle_index": cycle}
        next_design, context, capacity_patch = _prepare_next_design(
            run=run_context, dataset=analysis_dataset, goal=goal, metrics=metrics, evidence=evidence or [], excluded_conditions=excluded,
            candidate_pool=[
                batch.get("condition") or {}
                for batch in dataset.get("batches") or []
                if batch.get("id") in set(replay_split.get("hidden_batch_ids") or [])
            ] if replay_split else None,
        )
        if capacity_patch:
            _artifact(runtime, "plan_patch", capacity_patch.to_dict())
        if next_design is not None and context.get("validation", {}).get("valid"):
            _artifact(runtime, "experiment_design", next_design.to_dict())
            next_pending = create_experiment_proposal(
                scientific_run_id, next_design, agent_run_id=run.get("agent_run_id"), replay_split=replay_split,
            )
    if next_pending is None:
        scientific_db.update_scientific_run(scientific_run_id, status="succeeded", cycle_index=cycle)
        if run.get("agent_run_id"):
            finish_run(
                run["agent_run_id"], status="succeeded", final_route="scientific_task", risk_level="medium",
                response_summary="Scientific digital-twin loop completed with simulation-only provenance.",
            )
    response = {
        "status": "success",
        "scientific_run_id": scientific_run_id,
        "pending_id": pending_id,
        "idempotency_key": idempotency_key,
        "simulation": {
            "simulation_id": result_id,
            "simulation_only": True,
            "provenance": simulation["provenance"],
            "measurement_count": len(simulation["measurements"]),
        },
        "cycle_index": cycle,
        "next_pending": next_pending,
    }
    if finalize_pending:
        pending_actions.mark_pending_executed(pending_id, response)
    return response
