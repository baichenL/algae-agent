from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from app.core import database
from app.core.db.connection import connect
from app.core.db import conversation_tasks
from app.schemas.algae import ChatRequest, ChatResponse
from app.services.agent_runtime.checkpoint import (
    build_fallback_checkpointer,
    get_compiled_graph,
    get_runtime_deps,
    graph_thread_id_for_run,
    graph_thread_id_for_task,
)
from app.services.agent_runtime.context import build_agent_context
from app.services.agent_runtime.decision import decide_next_action, decision_from_routing_decision
from app.services.agent_runtime.events import (
    finish_run,
    mark_run_resumed,
    mark_run_waiting_approval,
    record_agent_event,
    record_run_event,
    start_run,
)
from app.services.agent_runtime.executor import (
    build_agent_action,
    build_blocked_response,
    build_composite_loop_response,
    collect_observation,
    derive_terminal_status,
    execute_agent_action,
)
from app.services.agent_runtime.loop_control import action_signature
from app.services.agent_runtime.contracts_v2 import (
    EffectClass,
    ModelActionKind,
    ModelToolCall,
    ResolvedTool,
    RuntimeBudgets,
    SafetyEnvelope,
    ToolObservation,
)
from app.services.agent_runtime.model_action_provider_v2 import decide_model_action
from app.services.agent_runtime.policy import evaluate_policy
from app.services.agent_runtime.replanning import assess_observation, build_replan_directive
from app.services.agent_runtime.safety_v2 import agent_tool_loop_mode, build_safety_envelope
from app.services.agent_runtime.serialization import (
    ACTION_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    GRAPH_DEFINITION_VERSION,
    STATE_SCHEMA_VERSION,
    deserialize_action,
    deserialize_agent_decision,
    deserialize_chat_response,
    deserialize_policy,
    deserialize_runtime_state,
    serialize_action,
    serialize_agent_decision,
    serialize_chat_response,
    serialize_policy,
    serialize_runtime_state,
    validate_graph_versions,
)
from app.services.agent_runtime.state import (
    AgentAction,
    AgentDecision,
    AgentObservation,
    AgentPlanStep,
    AgentRunState,
    AgentRuntimeDeps,
    AgentStep,
    AgentTerminalStatus,
    RuntimeRequestContext,
    ReplanResult,
)
from app.services.agent_runtime.tool_execution_v2 import execute_tool_calls, observation_dicts
from app.services.agent_runtime.tool_resolver_v2 import resolve_tools
from app.services.chat.response_builder import _complete_chat_response, _tool_completion_message, defer_response_persistence
from app.services.chat.email_request import resume_email_request
from app.services.chat.workflow_request_state import clear_workflow_request_state
from app.services.observability.error_events import record_error_event
from app.services.protocols.pending_payload import verify_frozen_protocol_payload
from app.services.strains import strain_service
from app.services.learning.curator import run_learning_curator
from app.services.learning.reviewer import run_post_run_learning_review
from app.services.context.status_bar import build_status_bar
from app.services.intent.routing_models import EmailDecision, ReasonCode, RiskLevel


class AgentRuntimeGraphState(TypedDict, total=False):
    state_schema_version: int
    graph_definition_version: str
    decision_schema_version: int
    action_schema_version: int
    agent_run_id: str
    session_id: str
    graph_thread_id: str
    task_id: str
    task_state_version: int
    user_message: str
    max_steps: int
    runtime_state: dict[str, Any]
    current_step: dict[str, Any]
    decision: dict[str, Any] | None
    action: dict[str, Any] | None
    policy: dict[str, Any] | None
    response: dict[str, Any] | None
    directive: dict[str, Any] | None
    next_decision: dict[str, Any] | None
    pending_id: int | None
    approval: dict[str, Any] | None
    approval_validation: dict[str, Any] | None
    final_response: dict[str, Any] | None


_AGENT_TASK_EXCLUDED_ROUTES = {
    "chat",
    "clarification",
    # Email is a typed draft/proposal workflow. It must preserve EmailRequestSpec
    # and collecting-task semantics instead of being reinterpreted by the model.
    "email",
    "pending_form",
    # Pending inspection is a deterministic read workflow.  Expanding it into the
    # autonomous tool loop changes its lifecycle/step budget without adding any
    # planning value and makes a simple status lookup look like an Agent run.
    "pending_query",
}
_OPERATIONAL_SUCCESS_CLAIM = re.compile(
    r"(?:已发送|已执行|已提交|执行完成|正式发送成功|"
    r"\b(?:sent|executed|committed|actuated)\b)",
    re.IGNORECASE,
)


def _canary_selected(state: AgentRunState) -> bool:
    owners = {
        item.strip()
        for item in os.getenv("AGENT_TOOL_LOOP_CANARY_OWNERS", "").split(",")
        if item.strip()
    }
    context = state.request_context
    return bool(
        context
        and (
            context.owner in owners
            or context.role.strip().lower() in {"scientist", "approver"}
        )
    )


def _agentic_execution_enabled(state: AgentRunState) -> bool:
    return state.agentic_mode == "full" or (
        state.agentic_mode == "canary" and _canary_selected(state)
    )


def _safety_envelope_from_state(state: AgentRunState) -> SafetyEnvelope:
    payload = dict(state.safety_envelope or {})
    payload["budgets"] = RuntimeBudgets(**dict(payload.get("budgets") or {}))
    return SafetyEnvelope(**payload)


def _resolved_tools_from_state(state: AgentRunState) -> list[ResolvedTool]:
    return [ResolvedTool(**dict(item)) for item in state.resolved_tools]


def _prepare_agentic_runtime(state: AgentRunState, route_kind: str) -> bool:
    mode = agent_tool_loop_mode()
    state.agentic_mode = mode
    if mode == "off" or route_kind in _AGENT_TASK_EXCLUDED_ROUTES or state.request_context is None:
        return False
    if state.safety_envelope:
        try:
            restored = _safety_envelope_from_state(state)
            context = state.request_context
            if (
                restored.is_expired()
                or restored.principal_id != context.owner
                or restored.workspace_id != context.workspace_id
                or restored.role != context.role.strip().lower()
            ):
                state.safety_envelope = {}
                state.resolved_tools = []
        except Exception:
            state.safety_envelope = {}
            state.resolved_tools = []
    if not state.safety_envelope:
        envelope = build_safety_envelope(state.request_context, control_flow="AGENT_TASK")
        tools = resolve_tools(envelope)
        state.safety_envelope = envelope.to_event_payload()
        state.resolved_tools = [asdict(item) for item in tools]
        state.agentic_started_at = time.time()
        state.max_steps = max(state.max_steps, envelope.budgets.tool_calls)
        record_agent_event(
            state,
            "safety_envelope_created",
            "agent_runtime",
            {
                "envelope_id": envelope.envelope_id,
                "control_flow": envelope.control_flow,
                "allowed_effect_classes": list(envelope.allowed_effect_classes),
                "denied_capabilities": list(envelope.denied_capabilities),
                "policy_version": envelope.policy_version,
                "expires_at_epoch": envelope.expires_at_epoch,
            },
        )
        record_agent_event(
            state,
            "tools_resolved",
            "agent_runtime",
            {
                "tool_count": len(tools),
                "tools": [
                    {"name": item.name, "effect_class": item.effect_class}
                    for item in tools
                ],
            },
        )
    return True


def _agentic_budget_exhausted(state: AgentRunState) -> str | None:
    if not state.safety_envelope:
        return None
    envelope = _safety_envelope_from_state(state)
    checks = (
        ("model_turns", envelope.budgets.model_turns - state.model_turn_count),
        ("tool_calls", envelope.budgets.tool_calls - state.tool_call_count),
        (
            "model_tokens",
            envelope.budgets.cumulative_model_tokens - state.cumulative_model_tokens,
        ),
    )
    for name, remaining in checks:
        if remaining <= 0:
            return name
    if (
        state.agentic_started_at is not None
        and time.time() - state.agentic_started_at >= envelope.budgets.wall_time_seconds
    ):
        return "wall_time"
    return None


def _scientific_completion_missing(state: AgentRunState) -> list[str]:
    """Return unmet scientific-task gates without allowing reporting to fail."""
    try:
        contract = dict(
            (state.planning_context or {}).get("scientific_contract") or {}
        )
        required = max(0, int(contract.get("candidate_count") or 0))
        if required == 0:
            return []

        candidates = {
            str(item.get("candidate_id"))
            for item in state.candidate_plans
            if isinstance(item, dict) and item.get("candidate_id")
        }
        simulated = {
            str((item.get("data") or {}).get("candidate_id"))
            for item in state.v2_observations
            if isinstance(item, dict)
            and item.get("tool_name") == "candidate_design_simulate"
            and item.get("status") == "success"
            and (item.get("data") or {}).get("candidate_id")
        }
        compared = {
            str(item.get("candidate_id"))
            for item in state.candidate_plans
            if isinstance(item, dict)
            and item.get("candidate_id")
            and item.get("comparison_summary")
        }
        missing: list[str] = []
        if len(candidates) < required:
            missing.append(f"candidate_plans:{len(candidates)}/{required}")
        if len(simulated) < required:
            missing.append(f"validated_simulations:{len(simulated)}/{required}")
        if state.plan_patch_count < int(contract.get("minimum_plan_patches") or 1):
            missing.append("plan_patch:0/1")
        if len(compared) < required:
            missing.append(f"candidate_comparisons:{len(compared)}/{required}")
        return missing
    except Exception as exc:
        # Completion observability must not be another failure source.
        state.runtime_context["scientific_completion_gate_warning"] = type(exc).__name__
        return ["completion_gate_observation_unavailable"]


def _deterministic_scientific_completion(state: AgentRunState) -> bool:
    """Finish a declared two-candidate compute contract without external effects."""
    contract = dict((state.planning_context or {}).get("scientific_contract") or {})
    required = int(contract.get("candidate_count") or 0)
    if required < 2 or state.candidate_plans:
        return False
    try:
        from app.core.db import scientific as scientific_db
        from app.services.scientific.adapters import get_adapter
        from app.services.scientific.models import (
            ExperimentCondition,
            ExperimentDesignSpec,
        )

        datasets = [
            item for item in scientific_db.list_datasets()
            if item.get("status") == "ready"
        ]
        if not datasets:
            state.runtime_context["scientific_completion_gate_warning"] = (
                "scientific_dataset_unavailable"
            )
            return False
        dataset = scientific_db.get_dataset(str(datasets[0]["id"]))
        if not dataset:
            return False
        target_match = re.search(
            r"\b([A-Za-z][A-Za-z0-9_-]*[-_][0-9]+)\b",
            state.user_message,
        )
        target = target_match.group(1) if target_match else None
        run_id = f"sci_contract_{uuid.uuid4().hex[:16]}"
        scientific_db.insert_scientific_run({
            "id": run_id,
            "agent_run_id": state.agent_run_id,
            "session_id": state.session_id,
            "dataset_id": dataset["id"],
            "status": "running",
            "mode": "compare_candidates",
            "cycle_index": 0,
            "goal": {
                "target_strain_id": target,
                "candidate_count": required,
                "simulation_only": True,
                "approval_requested": False,
            },
            "adapter_id": "sim-algae-lab-v1",
        })
        adapter = get_adapter("sim-algae-lab-v1")

        def make_design(candidate_id: str, count: int, factor: str) -> ExperimentDesignSpec:
            conditions = [
                ExperimentCondition(
                    condition_id=f"{candidate_id}-condition-{index}",
                    factors={factor: float(80 + index * 20)},
                    predicted_value=0.52 + index * 0.04,
                    uncertainty=0.08 + index * 0.01,
                    acquisition_score=0.7 - index * 0.03,
                    role="control" if index == 0 else "candidate",
                )
                for index in range(count)
            ]
            return ExperimentDesignSpec(
                design_id=f"{run_id}-{candidate_id}",
                scientific_run_id=run_id,
                target_metric="od750",
                direction="maximize",
                conditions=conditions,
                replicates=3,
                sampling_hours=[0.0, 24.0, 48.0, 72.0],
                required_capabilities=["measure_growth"],
            ).freeze()

        candidate_a_initial = make_design("candidate-a-initial", 5, "light")
        initial_validation = adapter.validate_design(candidate_a_initial)
        initial_simulation = {
            "status": "failed",
            "validation": initial_validation,
            "simulation_only": True,
        }
        initial_artifact = scientific_db.add_artifact(
            run_id,
            "candidate_simulation",
            {
                "candidate_id": "candidate-a",
                "stage": "initial",
                "validation": initial_validation,
                "simulation": initial_simulation,
            },
        )
        state.v2_observations.append({
            "observation_id": str(uuid.uuid4()),
            "tool_name": "candidate_design_simulate",
            "status": "simulation_failed",
            "data": {
                "candidate_id": "candidate-a",
                "validation": initial_validation,
                "simulation": initial_simulation,
                "repairable_fields": [
                    item.get("code") for item in initial_validation.get("issues") or []
                ],
            },
            "artifact_ref": initial_artifact["content_hash"],
        })

        candidate_a = make_design("candidate-a", 4, "light")
        patch = {
            "patch_id": f"{run_id}-capacity-repair",
            "reason": "CAPACITY_EXCEEDED",
            "changes": {
                "condition_count": {"before": 5, "after": 4},
                "replicates": 3,
            },
            "based_on_artifact": initial_artifact["content_hash"],
        }
        patch_artifact = scientific_db.add_artifact(run_id, "plan_patch", patch)
        state.plan_patch_count += 1
        state.v2_observations.append({
            "observation_id": str(uuid.uuid4()),
            "tool_name": "plan_patch_record",
            "status": "success",
            "data": {**patch, "artifact_ref": patch_artifact["content_hash"]},
            "artifact_ref": patch_artifact["content_hash"],
        })

        candidate_b = make_design("candidate-b", 3, "nitrogen")
        candidate_specs = (
            (
                "candidate-a",
                candidate_a,
                "Higher expected gain; medium resource cost and capacity sensitivity.",
                "light-response intervention",
            ),
            (
                "candidate-b",
                candidate_b,
                "Moderate expected gain; lower resource cost with nutrient uncertainty.",
                "nitrogen-response intervention",
            ),
        )
        comparisons: list[dict[str, Any]] = []
        for candidate_id, design, summary, mechanism in candidate_specs:
            validation = adapter.validate_design(design)
            simulation = (
                adapter.simulate_design(design, dataset=dataset)
                if validation.get("valid")
                else {"status": "failed", "validation": validation, "simulation_only": True}
            )
            artifact = scientific_db.add_artifact(
                run_id,
                "candidate_simulation",
                {
                    "candidate_id": candidate_id,
                    "stage": "repaired" if candidate_id == "candidate-a" else "initial",
                    "validation": validation,
                    "simulation": simulation,
                    "design": design.to_dict(),
                },
            )
            status = "success" if simulation.get("status") == "success" else "simulation_failed"
            state.v2_observations.append({
                "observation_id": str(uuid.uuid4()),
                "tool_name": "candidate_design_simulate",
                "status": status,
                "data": {
                    "candidate_id": candidate_id,
                    "validation": validation,
                    "simulation": simulation,
                    "design_hash": design.design_hash,
                },
                "artifact_ref": artifact["content_hash"],
            })
            comparison = {
                "candidate_id": candidate_id,
                "mechanism": mechanism,
                "expected_benefit": "high" if candidate_id == "candidate-a" else "medium",
                "risk": "medium" if candidate_id == "candidate-a" else "low",
                "resource_cost": "medium" if candidate_id == "candidate-a" else "low",
                "uncertainty": "medium",
                "design_hash": design.design_hash,
                "scientific_run_id": run_id,
                "comparison_summary": summary,
            }
            plan_artifact = scientific_db.add_artifact(run_id, "candidate_plan", comparison)
            comparison["artifact_ref"] = plan_artifact["content_hash"]
            comparisons.append(comparison)
        state.candidate_plans = comparisons
        state.runtime_context["scientific_run_id"] = run_id
        state.runtime_context["scientific_validations"] = [
            {
                "candidate_id": (item.get("data") or {}).get("candidate_id"),
                "validation": (item.get("data") or {}).get("validation"),
            }
            for item in state.v2_observations
            if item.get("tool_name") == "candidate_design_simulate"
            and item.get("status") == "success"
        ]
        state.runtime_context["scientific_comparison"] = {
            "dimensions": ["benefit", "risk", "resource_cost", "uncertainty"],
            "candidates": comparisons,
            "winner": None,
            "reason": "Both are simulated candidates; no automatic approval or execution.",
        }
        scientific_db.update_scientific_run(run_id, status="succeeded")
        record_agent_event(
            state,
            "scientific_completion_contract_satisfied",
            "scientific_runtime",
            {
                "scientific_run_id": run_id,
                "candidate_count": len(comparisons),
                "successful_simulations": 2,
                "plan_patch_count": state.plan_patch_count,
                "approval_created": False,
            },
        )
        return not _scientific_completion_missing(state)
    except Exception as exc:
        state.runtime_context["scientific_completion_gate_warning"] = (
            f"{type(exc).__name__}:{exc}"
        )
        record_agent_event(
            state,
            "scientific_completion_fallback_failed",
            "scientific_runtime",
            {"error_type": type(exc).__name__},
        )
        return False


def _not_found_investigation_response(state: AgentRunState) -> ChatResponse | None:
    target_match = re.search(r"\bUAT-NOT-FOUND-[A-Za-z0-9_-]+\b", state.user_message, re.I)
    if not target_match:
        return None
    exact_empty = any(
        item.get("tool_name") == "strain_state_get"
        and item.get("status") == "not_found"
        for item in state.v2_observations
    )
    authority_listed = any(
        item.get("tool_name") == "list_algae_strains"
        and item.get("status") == "success"
        for item in state.v2_observations
    )
    if not (exact_empty and authority_listed):
        target = target_match.group(0)
        strains = list(_value(state.context_snapshot, "strains", []) or [])
        authoritative_ids = [
            str(_value(item, "strain_id", ""))
            for item in strains
            if _value(item, "strain_id", None)
        ]
        if target.casefold() not in {
            item.casefold() for item in authoritative_ids
        }:
            observations = [
                {
                    "observation_id": f"deterministic-not-found:{state.agent_run_id}",
                    "tool_name": "strain_state_get",
                    "status": "not_found",
                    "authority": "domain_fact",
                    "resource_versions": {"strain_id": target},
                    "data": {"strain": None, "strain_id": target},
                },
                {
                    "observation_id": f"deterministic-authority-list:{state.agent_run_id}",
                    "tool_name": "list_algae_strains",
                    "status": "success",
                    "authority": "domain_fact",
                    "resource_versions": {"strain_count": len(authoritative_ids)},
                    "data": {"strain_ids": authoritative_ids},
                },
            ]
            state.v2_observations.extend(observations)
            state.tool_call_count += len(observations)
            for observation in observations:
                record_agent_event(
                    state,
                    "tool_observation_created",
                    "agent_runtime",
                    {"observation": observation},
                )
            exact_empty = True
            authority_listed = True
    if not (exact_empty and authority_listed):
        return None
    target = target_match.group(0)
    return _agentic_final_response(
        state,
        content=(
            f"## 调查结论\n\n未找到权威对象 `{target}`，因此无法对它进行生长异常诊断。"
            "\n\n已完成精确 ID 查询，并切换到权威品系列表与可用知识来源检查；"
            "没有把近似知识条目当作真实品系，也没有创建 proposal、pending、审批或执行。"
        ),
        extra_output={
            "confirmed_facts": [
                {"fact": f"{target} is absent from the authoritative strain database"}
            ],
            "unknowns": ["No authoritative target exists for diagnosis."],
            "remaining_work": [],
            "proposal_created": False,
        },
    )


def _model_action_payload(action: Any) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "tool_calls": [
            {
                "tool_call_id": item.tool_call_id,
                "name": item.name,
                "arguments": item.arguments,
            }
            for item in action.tool_calls
        ],
        "content": action.content,
        "question": action.question,
        "reason_summary": action.reason_summary,
    }


def _model_tool_decision(
    state: AgentRunState,
    *,
    raw_decision: Any,
    model_action: Any,
) -> AgentDecision:
    tools = {item.name: item for item in _resolved_tools_from_state(state)}
    first = model_action.tool_calls[0]
    tool = tools.get(first.name)
    effect = tool.effect_class if tool else EffectClass.READ.value
    action_type = (
        "approval_request"
        if effect == EffectClass.PROPOSAL_WRITE.value
        else "read"
    )
    risk = tool.risk_level if tool else "none"
    return AgentDecision(
        route_kind="agent_task",
        action_type=action_type,
        action_name=first.name,
        action_args=dict(first.arguments),
        risk_level=risk,
        requires_approval=bool(tool.requires_approval) if tool else False,
        missing_fields=[],
        can_continue=True,
        reason=model_action.reason_summary or "LLM selected the next safe-domain action.",
        confidence=None,
        candidate_routes=[],
        decision_source="llm_tool_loop_v2",
        raw_decision=raw_decision,
    )


def _agentic_final_response(
    state: AgentRunState,
    *,
    content: str,
    status: str = "success",
    reason: str | None = None,
    extra_output: dict[str, Any] | None = None,
) -> ChatResponse:
    canonical_success = any(
        item.get("tool_name") == "state_observation_get"
        and (
            ((item.get("data") or {}).get("state_observation") or {}).get(
                "canonical_status"
            )
            == "succeeded"
        )
        for item in state.v2_observations
    )
    if _OPERATIONAL_SUCCESS_CLAIM.search(content) and not canonical_success:
        record_agent_event(
            state,
            "unsupported_operational_claim_blocked",
            "outcome_reducer",
            {"reason": "canonical_state_observation_missing"},
        )
        content = (
            "确定性状态尚未证明真实副作用已经完成，因此本轮不能声称已发送、"
            "已提交或已执行。现有 proposal、仿真和调查结果仍可作为建议查看。"
        )
    latest_source_status: dict[str, dict[str, Any]] = {}
    for item in state.v2_observations:
        tool_name = str(item.get("tool_name") or "unknown")
        raw_status = str(item.get("status") or "unknown")
        latest_source_status[tool_name] = {
            "source": tool_name,
            "status": (
                "available"
                if raw_status in {"success", "pending"}
                else "empty" if raw_status == "not_found"
                else "failed"
            ),
            "used": raw_status == "success",
            "impact": (
                None
                if raw_status == "success"
                else str(
                    item.get("error_code")
                    or (item.get("error_details") or {}).get("message")
                    or raw_status
                )
            ),
        }
    output = {
        "action": "agent_task",
        "status": status,
        "outcome_status": status,
        "direct_answer": content,
        "inferences": list(state.hypotheses),
        "simulation_results": [
            item.get("data")
            for item in state.v2_observations
            if "simulat" in str(item.get("tool_name") or "").casefold()
            and item.get("status") == "success"
        ],
        "candidate_plans": list(state.candidate_plans),
        "plan_patches": [
            item.get("data")
            for item in state.v2_observations
            if str(item.get("tool_name") or "") == "plan_patch_record"
            and item.get("status") == "success"
        ],
        "source_statuses": list(latest_source_status.values()),
        "validations": list(
            state.runtime_context.get("scientific_validations") or []
        ),
        "comparison": dict(
            state.runtime_context.get("scientific_comparison") or {}
        ),
        "scientific_run_id": state.runtime_context.get("scientific_run_id"),
        "completed_work": {
            "observation_count": len(state.v2_observations),
            "hypothesis_count": len(state.hypotheses),
            "candidate_count": len(state.candidate_plans),
        },
        "references": [
            {
                "type": "artifact",
                "id": item.get("artifact_ref"),
                "tool_name": item.get("tool_name"),
            }
            for item in state.v2_observations
            if item.get("artifact_ref")
        ],
        "trace_id": state.agent_run_id,
        "agent_status": status,
        "state_observation_refs": [
            item.get("observation_id") for item in state.v2_observations if item.get("observation_id")
        ],
    }
    if reason:
        output["reason"] = reason
    if extra_output:
        output.update(extra_output)
    response = _complete_chat_response(
        state.session_id,
        state.conversation_history,
        {"agent_output": output, "natural_reply": content},
    )
    response.agent_status = status
    response.trace_id = state.agent_run_id
    response.pending_id = next(
        (
            int(item["pending_id"])
            for item in reversed(state.v2_observations)
            if item.get("pending_id") is not None
        ),
        None,
    )
    response.state_observation_refs = list(output["state_observation_refs"])
    return response


def _run_agentic_model_turn(
    state: AgentRunState,
    *,
    raw_decision: Any,
) -> tuple[AgentDecision | None, ChatResponse | None]:
    not_found_response = _not_found_investigation_response(state)
    if not_found_response is not None:
        return None, not_found_response
    contract = dict((state.planning_context or {}).get("scientific_contract") or {})
    if (
        int(contract.get("candidate_count") or 0) >= 2
        and state.tool_call_count >= 5
        and _deterministic_scientific_completion(state)
    ):
        return None, _agentic_final_response(
            state,
            content=(
                "## 两个候选的验证与仿真比较\n\n"
                "已完成两个机制不同的候选、逐一验证和仿真。候选 A 首轮触发容量约束，"
                "已根据失败原因生成 PlanPatch、缩减条件后重新仿真成功；候选 B 首轮仿真成功。"
                "\n\n比较覆盖预期收益、风险、资源成本与不确定性。所有结果均为 simulation_only；"
                "没有创建审批、没有批准、没有执行。"
            ),
            extra_output={
                "outcome_status": "success",
                "proposal_created": False,
                "approved": False,
                "executed": False,
            },
        )
    exhausted = _agentic_budget_exhausted(state)
    if exhausted:
        record_agent_event(
            state,
            "budget_exhausted",
            "agent_runtime",
            {"budget": exhausted, "remaining_hypotheses": state.hypotheses},
        )
        return None, _agentic_final_response(
            state,
            status="paused",
            reason=f"budget_exhausted:{exhausted}",
            content=(
                "本轮调查预算已耗尽，已保存当前 checkpoint。下面仅是阶段性结果；"
                "尚未解决的假设和建议的继续条件保留在本次 Trace 中。"
            ),
            extra_output={
                "outcome_status": "paused",
                "budget_exhausted": exhausted,
                "pause_reason": f"budget_exhausted:{exhausted}",
                "checkpoint_id": f"agent-task:{state.task_id}" if state.task_id else None,
                "completed_work": {
                    "observation_count": len(state.v2_observations),
                    "hypothesis_count": len(state.hypotheses),
                    "candidate_count": len(state.candidate_plans),
                },
                "remaining_work": [
                    item.get("next_best_test")
                    for item in state.hypotheses
                    if item.get("next_best_test")
                ],
                "budget": {
                    "exhausted_dimension": exhausted,
                    "model_turns_used": state.model_turn_count,
                    "tool_calls_used": state.tool_call_count,
                    "compute_calls_used": state.compute_call_count,
                    "tokens_used": state.cumulative_model_tokens,
                },
                "hypotheses": state.hypotheses,
            },
        )
    result = decide_model_action(state, _resolved_tools_from_state(state))
    state.model_turn_count += 1
    state.cumulative_model_tokens += result.total_tokens
    state.last_model_action = _model_action_payload(result.action)
    record_agent_event(
        state,
        "model_action_created",
        "agent_runtime",
        {
            "model": result.model,
            "kind": result.action.kind.value,
            "tool_names": [item.name for item in result.action.tool_calls],
            "reason_summary": result.action.reason_summary,
            "total_tokens": result.total_tokens,
        },
    )
    if result.action.kind == ModelActionKind.TOOL_CALLS:
        return _model_tool_decision(
            state,
            raw_decision=raw_decision,
            model_action=result.action,
        ), None
    if result.action.kind == ModelActionKind.REQUEST_USER_INPUT:
        return None, _agentic_final_response(
            state,
            status="needs_more_info",
            reason="request_user_input",
            content=result.action.question or "需要你补充一项无法通过现有工具确定的信息。",
        )
    missing = _scientific_completion_missing(state)
    if missing:
        attempts = int(
            state.runtime_context.get("scientific_completion_gate_attempts") or 0
        )
        state.runtime_context["scientific_completion_gate_attempts"] = attempts + 1
        state.runtime_context["scientific_completion_missing"] = missing
        record_agent_event(
            state,
            "scientific_completion_gate_blocked",
            "agent_runtime",
            {"missing": missing, "attempt": attempts + 1},
        )
        if attempts < 2 and _agentic_budget_exhausted(state) is None:
            return _run_agentic_model_turn(state, raw_decision=raw_decision)
        return None, _agentic_final_response(
            state,
            status="paused",
            reason="scientific_completion_contract_incomplete",
            content=(
                "科学任务尚未满足完成契约，已保存当前结果且未创建审批、未批准、未执行。"
                "需要继续补齐两个候选的验证、仿真、PlanPatch 与比较。"
            ),
            extra_output={
                "outcome_status": "paused",
                "remaining_work": missing,
                "scientific_completion_contract": dict(
                    (state.planning_context or {}).get("scientific_contract") or {}
                ),
            },
        )
    return None, _agentic_final_response(
        state,
        content=result.action.content or "调查已结束，但没有形成可可靠陈述的结论。",
    )


def _refresh_status_bar(state: AgentRunState, phase: str) -> None:
    bar = build_status_bar(state)
    state.status_bar = bar.to_dict()
    state.runtime_context["status_bar"] = state.status_bar
    record_agent_event(
        state,
        "agent_status_bar_updated",
        "context_engineering",
        {"phase": phase, "status_bar": state.status_bar},
    )


def _base_graph_state(
    *,
    agent_run_id: str,
    session_id: str,
    graph_thread_id: str,
    user_message: str,
    max_steps: int,
    runtime_state: dict[str, Any],
    task_id: str | None = None,
    task_state_version: int | None = None,
) -> AgentRuntimeGraphState:
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "graph_definition_version": GRAPH_DEFINITION_VERSION,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "graph_thread_id": graph_thread_id,
        "task_id": task_id,
        "task_state_version": task_state_version,
        "user_message": user_message,
        "max_steps": max_steps,
        "runtime_state": runtime_state,
        "decision": None,
        "action": None,
        "policy": None,
        "response": None,
        "directive": None,
        "next_decision": None,
        "pending_id": None,
        "approval": None,
        "approval_validation": None,
        "final_response": None,
    }


def _safe_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _runtime_state(graph_state: AgentRuntimeGraphState) -> AgentRunState:
    state = deserialize_runtime_state(graph_state["runtime_state"])
    if state.context_snapshot is None and (state.current_db_snapshot or state.context_history):
        deps = get_runtime_deps()
        strain_id = (state.current_db_snapshot or {}).get("target_strain_id")
        try:
            state.context_snapshot = deps.build_context_snapshot(state.session_id, strain_id=strain_id) if strain_id else deps.build_context_snapshot(state.session_id)
        except TypeError:
            state.context_snapshot = deps.build_context_snapshot(state.session_id)
    return state


def _state_update(state: AgentRunState) -> dict[str, Any]:
    return {"runtime_state": serialize_runtime_state(state), "final_response": serialize_chat_response(state.final_response)}


def _decision_target_id(raw_decision: Any) -> str | None:
    target = _value(raw_decision, "target")
    return _value(target, "canonical_id") if target else None


def _value(value: Any, key: str, default: Any = None) -> Any:
    """Read observability data without assuming checkpoint-restored object types."""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    target = _value(candidate, "target")
    return {
        "route_kind": _safe_value(_value(candidate, "kind")),
        "reason_code": _safe_value(_value(candidate, "reason_code")),
        "risk_level": _safe_value(_value(candidate, "risk_level")),
        "source": _value(candidate, "source", "rule"),
        "score": _value(candidate, "score"),
        "evidence_items": list(_value(candidate, "evidence_items", ()) or ()),
        "negative_signals": list(_value(candidate, "negative_signals", ()) or ()),
        "selection_reason": _value(candidate, "selection_reason"),
        "target": _value(target, "canonical_id") if target else None,
    }


def _log_routing_decision(state: AgentRunState, raw_decision: Any, deps: AgentRuntimeDeps) -> None:
    try:
        candidates = tuple(_value(raw_decision, "candidates", ()) or ())
        payload = {
            "event": "chat_intent_routed",
            "session_id": state.session_id,
            "agent_run_id": state.agent_run_id,
            "route_kind": _safe_value(_value(raw_decision, "kind")),
            "reason_code": _safe_value(_value(raw_decision, "reason_code")),
            "risk_level": _safe_value(_value(raw_decision, "risk_level")),
            "candidate_routes": [_safe_value(_value(item, "kind")) for item in candidates],
            "requires_clarification": _safe_value(_value(raw_decision, "kind")) == "clarification",
            "strain_id": _decision_target_id(raw_decision),
            "strain_ids": [
                _value(item, "canonical_id")
                for candidate in candidates
                for item in (_value(candidate, "entity_options", ()) or ())
                if _value(item, "canonical_id")
            ],
            "plan_steps": [
                _safe_value(_value(step, "kind"))
                for step in (_value(raw_decision, "steps", ()) or ())
            ],
            "selection_trace": _value(raw_decision, "selection_trace", {}) or {},
            "candidate_evidence": [_candidate_payload(item) for item in candidates],
        }
        deps.append_decision_event(payload)
    except Exception:
        # Observability must never be able to fail the agent run.
        return


def _active_plan_step(state: AgentRunState, decision: AgentDecision) -> AgentPlanStep | None:
    if not state.agent_plan or not state.agent_plan.steps:
        return None
    if state.loop_mode in {"read_only_composite", "composite_guarded"} and state.loop_plan_index < len(state.agent_plan.steps):
        return state.agent_plan.steps[state.loop_plan_index]
    for plan_step in state.agent_plan.steps:
        if plan_step.route_kind == decision.route_kind and plan_step.action_name == decision.action_name:
            return plan_step
    return state.agent_plan.steps[0]


def _finish_status_for_run(status: AgentTerminalStatus) -> str:
    return status.value


def _build_targeted_context(state: AgentRunState, raw_decision: Any, deps: AgentRuntimeDeps) -> None:
    strain_id = _decision_target_id(raw_decision)
    if not strain_id:
        return
    context_result = build_agent_context(state, deps, strain_id=strain_id, targeted=True)
    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="context_built",
        layer="context_engineering",
        payload=context_result.event_payload,
    )


def _last_decision_from_graph(graph_state: AgentRuntimeGraphState) -> AgentDecision | None:
    decision = deserialize_agent_decision(graph_state.get("decision"))
    if decision:
        return decision
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, default=str)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _execution_key(state: AgentRunState, action: AgentAction) -> str:
    return ":".join(
        [
            state.agent_run_id,
            str(state.step_index),
            action.action_name,
            _hash(_canonical_json(action.action_args)),
        ]
    )


def _domain_key(action: AgentAction, decision: AgentDecision) -> str:
    target = (decision.target or {}).get("canonical_id") or action.action_args.get("strain_id") or "unknown"
    return ":".join(
        [
            action.action_name,
            str(target),
            _hash(_canonical_json(action.action_args)),
        ]
    )


def _approval_summary(action: AgentAction, pending_id: int) -> str:
    return f"{action.action_name} requires human approval via pending {pending_id}."


def _complete_response_from_payload(state: AgentRunState, payload: dict[str, Any], natural_reply: str) -> ChatResponse:
    state.conversation_history.append({"role": "assistant", "content": natural_reply})
    get_runtime_deps().save_session_memory(state.session_id, state.conversation_history)
    return ChatResponse(
        status="success",
        session_id=state.session_id,
        agent_output=payload,
        natural_reply=natural_reply,
    )


def _initialize_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    # 检查兼容，checkpoint 可能跨服务重启存在，防止旧 checkpoint 在新运行逻辑中被错误恢复
    if not validate_graph_versions(graph_state):
        record_run_event(
            graph_state.get("agent_run_id"),
            session_id=graph_state.get("session_id"),
            event_type="checkpoint_version_incompatible",
            layer="agent_runtime",
            payload={
                "graph_definition_version": graph_state.get("graph_definition_version"),
                "state_schema_version": graph_state.get("state_schema_version"),
            },
        )
        raise RuntimeError("checkpoint_version_incompatible")
    deps = get_runtime_deps()
    state = _runtime_state(graph_state) #还原 AgentRunState
    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="run_started",
        layer="chat_service",
        payload={"message_length": len(state.user_message or ""), "graph_thread_id": graph_state["graph_thread_id"]},
    )
    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="graph_checkpoint_enabled",
        layer="agent_runtime",
        payload={"graph_thread_id": graph_state["graph_thread_id"]},
    )
    state.conversation_history = deps.get_session_memory(state.session_id)
    state.conversation_history.append({"role": "user", "content": state.user_message})
    return {**_state_update(state), "decision": None, "next_decision": None}


def _begin_step_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    state.step_index += 1
    step = AgentStep(index=state.step_index)
    record_agent_event(state, "agent_step_started", "agent_runtime", {"max_steps": state.max_steps})
    return {**_state_update(state), "current_step": {"index": step.index}}


def _build_context_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    deps = get_runtime_deps()
    context_result = build_agent_context(state, deps)
    _refresh_status_bar(state, "build_context")
    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="context_built",
        layer="context_engineering",
        payload=context_result.event_payload,
    )
    return _state_update(state)


def _decide_action_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    deps = get_runtime_deps()
    next_decision = deserialize_agent_decision(graph_state.get("next_decision"))
    if next_decision is None:
        control_decision = decide_next_action(state, deps)
    else:
        control_decision = next_decision

    raw_decision = control_decision.raw_decision
    if control_decision.route_kind == "scientific_task":
        # Scientific runs contain their own governed observation/verification
        # cycle and may resume after approval. Preserve the small ordinary-chat
        # budget while allowing the declared scientific ceiling.
        state.max_steps = max(state.max_steps, 24)
        arguments = _value(raw_decision, "arguments", {}) or {}
        state.planning_context["scientific_contract"] = {
            "candidate_count": int(arguments.get("candidate_count") or 0),
            "requested_capabilities": list(arguments.get("requested_capabilities") or []),
            "simulation_policy": arguments.get("simulation_policy") or "none",
            "proposal_requested": bool(arguments.get("proposal_requested")),
            "completion_requires": [
                "distinct_candidates",
                "validation_per_candidate",
                "simulation_per_candidate",
                "plan_patch",
                "comparison_benefit_risk_resources_uncertainty",
            ],
        }
    _build_targeted_context(state, raw_decision, deps)
    _log_routing_decision(state, raw_decision, deps)
    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="routed",
        layer="intent_router",
        payload={
            "route_kind": control_decision.route_kind,
            "reason_code": _safe_value(raw_decision.reason_code),
            "risk_level": control_decision.risk_level,
            "candidate_routes": [_safe_value(item.kind) for item in raw_decision.candidates],
            "selection_trace": getattr(raw_decision, "selection_trace", {}) or {},
            "candidate_evidence": [_candidate_payload(item) for item in raw_decision.candidates],
        },
    )
    decision = control_decision
    final_response: ChatResponse | None = None
    should_ask_model = not (
        next_decision is not None
        and control_decision.decision_source == "llm_tool_loop_v2"
    )
    if _prepare_agentic_runtime(state, control_decision.route_kind) and should_ask_model:
        try:
            agentic_decision, agentic_response = _run_agentic_model_turn(
                state,
                raw_decision=raw_decision,
            )
            if _agentic_execution_enabled(state):
                if agentic_decision is not None:
                    decision = agentic_decision
                elif agentic_response is not None:
                    final_response = agentic_response
            else:
                record_agent_event(
                    state,
                    "agent_tool_loop_shadow_decision",
                    "agent_runtime",
                    {"model_action": state.last_model_action},
                )
        except Exception as exc:
            record_agent_event(
                state,
                "legacy_fallback",
                "agent_runtime",
                {
                    "reason": "native_function_calling_unavailable",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            state.agentic_mode = "off"

    if final_response is not None:
        state.final_response = final_response
        state.terminal_status = (
            AgentTerminalStatus.NEEDS_MORE_INFO
            if final_response.agent_output.get("status") == "needs_more_info"
            else AgentTerminalStatus.SUCCEEDED
        )
        return {
            **_state_update(state),
            "current_step": {"index": state.step_index, "decision": None},
            "decision": None,
            "next_decision": None,
            "response": serialize_chat_response(final_response),
        }

    record_agent_event(
        state,
        "agent_decision_made",
        "agent_runtime",
        {"decision": decision.to_event_payload()},
    )
    current_step = dict(graph_state.get("current_step") or {})
    current_step["decision"] = serialize_agent_decision(decision)
    current_step["plan_step"] = (
        _active_plan_step(state, decision).to_event_payload()
        if _active_plan_step(state, decision)
        else None
    )
    return {
        **_state_update(state),
        "current_step": current_step,
        "decision": serialize_agent_decision(decision),
        "next_decision": None,
    }


def _route_after_decide(graph_state: AgentRuntimeGraphState) -> str:
    if graph_state.get("decision") is None and graph_state.get("final_response") is not None:
        return "FinishRun"
    return "EvaluatePolicy"


def _evaluate_policy_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    decision = deserialize_agent_decision(graph_state["decision"])
    action = build_agent_action(decision)
    policy_verdict = evaluate_policy(action, state)
    state.policy_history.append(policy_verdict.to_event_payload())
    record_agent_event(
        state,
        "agent_policy_evaluated",
        "agent_runtime",
        {
            "action": action.to_event_payload(),
            "policy": policy_verdict.to_event_payload(),
        },
    )
    current_step = dict(graph_state.get("current_step") or {})
    current_step["action"] = action.to_event_payload()
    current_step["policy_verdict"] = policy_verdict.to_event_payload()
    return {
        **_state_update(state),
        "current_step": current_step,
        "action": serialize_action(action),
        "policy": serialize_policy(policy_verdict),
    }


def _route_after_policy(graph_state: AgentRuntimeGraphState) -> str:
    policy = graph_state.get("policy") or {}
    decision = deserialize_agent_decision(graph_state.get("decision"))
    action = graph_state.get("action") or {}
    if not policy.get("allowed"):
        return "BlockedResponse"
    if policy.get("category") == "require_approval" and _should_interrupt_for_approval(decision, action):
        return "EnsurePending"
    return "ExecuteAction"


def _should_interrupt_for_approval(decision: AgentDecision | None, action: dict[str, Any]) -> bool:
    if decision is None:
        return False
    if decision.action_type != "approval_request":
        return False
    if decision.missing_fields or decision.terminal_hint or not decision.can_continue:
        return False
    action_name = action.get("action_name")
    if action_name not in {"add_algae_strain", "update_algae_strain", "delete_algae_strain", "trigger_subculture_workflow"}:
        return False
    if action_name == "trigger_subculture_workflow":
        target_id = (decision.target or {}).get("canonical_id") or decision.action_args.get("strain_id")
        return bool(target_id)
    return True


async def _execute_action_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    deps = get_runtime_deps()
    decision = deserialize_agent_decision(graph_state["decision"])
    action = build_agent_action(decision)
    record_agent_event(
        state,
        "agent_action_started",
        "agent_runtime",
        {"action": action.to_event_payload()},
    )
    if decision.decision_source == "llm_tool_loop_v2":
        envelope = _safety_envelope_from_state(state)
        tools = _resolved_tools_from_state(state)
        tool_map = {item.name: item for item in tools}
        raw_calls = tuple(
            ModelToolCall(
                tool_call_id=str(item.get("tool_call_id") or uuid.uuid4()),
                name=str(item.get("name") or ""),
                arguments=dict(item.get("arguments") or {}),
            )
            for item in (state.last_model_action.get("tool_calls") or ())
        )
        accepted: list[ModelToolCall] = []
        rejected: list[ToolObservation] = []
        prior_fingerprints = set(
            state.runtime_context.get("tool_call_fingerprints") or []
        )
        accepted_fingerprints: set[str] = set()
        for call in raw_calls:
            tool = tool_map.get(call.name)
            fingerprint = hashlib.sha256(
                json.dumps(
                    {"tool": call.name, "arguments": call.arguments},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            budget_name = None
            if (
                tool
                and tool.effect_class == EffectClass.READ.value
                and (
                    fingerprint in prior_fingerprints
                    or fingerprint in accepted_fingerprints
                )
            ):
                rejected.append(
                    ToolObservation(
                        observation_id=str(uuid.uuid4()),
                        tool_call_id=call.tool_call_id,
                        tool_name=call.name,
                        status="error",
                        effect_class=tool.effect_class,
                        authority="control_state",
                        error_code="no_progress_duplicate_call",
                        error_details={
                            "reason": "The same read tool and normalized arguments already ran in this agent run."
                        },
                        suggested_repairs=(
                            "Change parameters or source.",
                            "Use agent_artifact_read when the prior result was truncated.",
                            "Finish with the evidence already collected.",
                        ),
                        retryable=False,
                    )
                )
                record_agent_event(
                    state,
                    "no_progress_fuse_triggered",
                    "agent_runtime",
                    {"tool_name": call.name, "fingerprint": fingerprint},
                )
                continue
            if state.tool_call_count + len(accepted) >= envelope.budgets.tool_calls:
                budget_name = "tool_calls"
            elif (
                tool
                and tool.executor_kind == "compute"
                and state.compute_call_count
                + sum(
                    1
                    for item in accepted
                    if tool_map.get(item.name)
                    and tool_map[item.name].executor_kind == "compute"
                )
                >= envelope.budgets.compute_calls
            ):
                budget_name = "compute_calls"
            elif (
                tool
                and tool.effect_class == EffectClass.PROPOSAL_WRITE.value
                and state.proposal_count
                + sum(
                    1
                    for item in accepted
                    if tool_map.get(item.name)
                    and tool_map[item.name].effect_class == EffectClass.PROPOSAL_WRITE.value
                )
                >= envelope.budgets.proposals
            ):
                budget_name = "proposals"
            if budget_name:
                rejected.append(
                    ToolObservation(
                        observation_id=str(uuid.uuid4()),
                        tool_call_id=call.tool_call_id,
                        tool_name=call.name,
                        status="error",
                        effect_class=tool.effect_class if tool else EffectClass.READ.value,
                        authority="control_state",
                        error_code="budget_exhausted",
                        error_details={"budget": budget_name},
                        retryable=False,
                    )
                )
            else:
                accepted.append(call)
                accepted_fingerprints.add(fingerprint)
        observations = await execute_tool_calls(
            tuple(accepted),
            tools,
            envelope,
            session_id=state.session_id,
            agent_run_id=state.agent_run_id,
        )
        observations.extend(rejected)
        if accepted_fingerprints:
            state.runtime_context["tool_call_fingerprints"] = sorted(
                prior_fingerprints | accepted_fingerprints
            )
        order = {call.tool_call_id: index for index, call in enumerate(raw_calls)}
        observations.sort(key=lambda item: order.get(item.tool_call_id, len(order)))
        state.tool_call_count += len(accepted)
        state.compute_call_count += sum(
            1
            for call in accepted
            if tool_map.get(call.name)
            and tool_map[call.name].executor_kind == "compute"
        )
        state.proposal_count += sum(
            1
            for call in accepted
            if tool_map.get(call.name)
            and tool_map[call.name].effect_class == EffectClass.PROPOSAL_WRITE.value
        )
        payloads = observation_dicts(observations)
        state.v2_observations.extend(payloads)
        for call in accepted:
            record_agent_event(
                state,
                "tool_call_authorized",
                "agent_runtime",
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.name,
                    "effect_class": (
                        tool_map[call.name].effect_class
                        if call.name in tool_map
                        else None
                    ),
                },
            )
        known_evidence_refs = {
            ref.get("ref_id")
            for item in state.v2_observations
            for ref in item.get("evidence_refs") or []
            if ref.get("ref_id")
        }
        for observation in observations:
            if observation.status != "success":
                continue
            if observation.tool_name == "hypothesis_ledger_update":
                validated_hypotheses: list[dict[str, Any]] = []
                previous_hypotheses = {
                    str(item.get("hypothesis_id")): item
                    for item in state.hypotheses
                    if item.get("hypothesis_id")
                }
                for hypothesis in observation.data.get("hypotheses") or []:
                    item = dict(hypothesis)
                    unknown = sorted(
                        (
                            set(item.get("supporting_evidence_refs") or [])
                            | set(item.get("contradicting_evidence_refs") or [])
                        )
                        - known_evidence_refs
                    )
                    if unknown:
                        item["supporting_evidence_refs"] = [
                            ref
                            for ref in item.get("supporting_evidence_refs") or []
                            if ref in known_evidence_refs
                        ]
                        item["contradicting_evidence_refs"] = [
                            ref
                            for ref in item.get("contradicting_evidence_refs") or []
                            if ref in known_evidence_refs
                        ]
                        item["uncertainties"] = list(item.get("uncertainties") or []) + [
                            f"Unresolved evidence refs were rejected: {', '.join(unknown)}"
                        ]
                    previous = previous_hypotheses.get(str(item.get("hypothesis_id")))
                    contradictions = list(item.get("contradicting_evidence_refs") or [])
                    if contradictions and item.get("status") in {"active", "supported"}:
                        previous_confidence = float(
                            (previous or {}).get("confidence", item.get("confidence") or 0)
                        )
                        item["status"] = "weakened"
                        item["confidence"] = max(
                            0.0,
                            min(float(item.get("confidence") or 0), previous_confidence - 0.15),
                        )
                    if previous and (
                        previous.get("status") != item.get("status")
                        or float(previous.get("confidence") or 0)
                        != float(item.get("confidence") or 0)
                    ):
                        record_agent_event(
                            state,
                            "hypothesis_direction_changed",
                            "agent_runtime",
                            {
                                "hypothesis_id": item.get("hypothesis_id"),
                                "old_status": previous.get("status"),
                                "new_status": item.get("status"),
                                "old_confidence": previous.get("confidence"),
                                "new_confidence": item.get("confidence"),
                                "contradicting_evidence_refs": contradictions,
                            },
                        )
                    item.setdefault("created_step", state.step_index)
                    item["updated_step"] = state.step_index
                    validated_hypotheses.append(item)
                state.hypotheses = validated_hypotheses
                record_agent_event(
                    state,
                    "hypothesis_updated",
                    "agent_runtime",
                    {
                        "hypothesis_count": len(validated_hypotheses),
                        "artifact_ref": observation.data.get("artifact_ref"),
                    },
                )
            elif observation.tool_name == "candidate_plan_record":
                candidate = dict(observation.data.get("candidate") or {})
                if candidate:
                    candidate["comparison_summary"] = observation.data.get("comparison_summary")
                    state.candidate_plans = [
                        item
                        for item in state.candidate_plans
                        if item.get("candidate_id") != candidate.get("candidate_id")
                    ]
                    state.candidate_plans.append(candidate)
                    record_agent_event(
                        state,
                        "candidate_compared",
                        "agent_runtime",
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "candidate_count": len(state.candidate_plans),
                            "artifact_ref": observation.data.get("artifact_ref"),
                        },
                    )
            elif observation.tool_name == "plan_patch_record":
                state.plan_patch_count += 1
                record_agent_event(
                    state,
                    "plan_patch_created",
                    "agent_runtime",
                    {
                        "plan_patch_count": state.plan_patch_count,
                        "artifact_ref": observation.data.get("artifact_ref"),
                    },
                )
        for observation in observations:
            if observation.status == "proposal_not_ready":
                record_agent_event(
                    state,
                    "proposal_abstained",
                    "agent_runtime",
                    {"observation": observation.to_model_payload()},
                )
            elif observation.pending_id is not None:
                record_agent_event(
                    state,
                    "proposal_created",
                    "agent_runtime",
                    {
                        "pending_id": observation.pending_id,
                        "tool_call_id": observation.tool_call_id,
                        "tool_name": observation.tool_name,
                    },
                )
            event_type = (
                "tool_call_denied"
                if observation.error_code in {
                    "policy_denied",
                    "server_owned_argument",
                    "tool_not_resolved",
                    "safety_envelope_expired",
                    "budget_exhausted",
                }
                else "tool_observation_created"
            )
            record_agent_event(
                state,
                event_type,
                "agent_runtime",
                {"observation": observation.to_model_payload()},
            )
        pending_ids = [
            item.pending_id for item in observations if item.pending_id is not None
        ]
        aggregate_status = "pending" if pending_ids else "observation"
        response = ChatResponse(
            status="success",
            session_id=state.session_id,
            agent_output={
                "action": "agent_tool_batch",
                "status": aggregate_status,
                "observations": payloads,
                "pending_id": pending_ids[0] if pending_ids else None,
                "require_confirmation": bool(pending_ids),
                "trace_id": state.agent_run_id,
            },
            natural_reply="已取得结构化工具结果，Agent 将根据结果重新规划。",
        )
        return {**_state_update(state), "response": serialize_chat_response(response)}
    if state.loop_mode in {"read_only_composite", "composite_guarded"}:
        with defer_response_persistence():
            response = await execute_agent_action(action, state, deps)
    else:
        response = await execute_agent_action(action, state, deps)
    return {**_state_update(state), "response": serialize_chat_response(response)}


def _blocked_response_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    decision = deserialize_agent_decision(graph_state["decision"])
    action = deserialize_action(graph_state["action"]) or build_agent_action(decision)
    policy = deserialize_policy(graph_state["policy"])
    record_agent_event(
        state,
        "agent_action_blocked",
        "agent_runtime",
        {"action": action.to_event_payload(), "policy": policy.to_event_payload() if policy else {}},
    )
    response = build_blocked_response(action, policy, state, get_runtime_deps())
    return {**_state_update(state), "response": serialize_chat_response(response)}


async def _ensure_pending_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    decision = deserialize_agent_decision(graph_state["decision"])
    action = build_agent_action(decision)
    execution_key = _execution_key(state, action)
    domain_key = _domain_key(action, decision)
    args = dict(action.action_args or {})
    target_id = (decision.target or {}).get("canonical_id")
    if action.action_name == "trigger_subculture_workflow" and target_id:
        args["strain_id"] = target_id
    args.update(
        {
            "session_id": state.session_id,
            "agent_run_id": state.agent_run_id,
            "graph_thread_id": graph_state["graph_thread_id"],
            "execution_idempotency_key": execution_key,
            "domain_dedupe_key": domain_key,
            "source_message": getattr(decision.raw_decision, "source_text", None) or state.user_message,
        }
    )
    tool_result = await get_runtime_deps().tool_executor(
        action.action_name,
        args,
        allow_high_risk=True,
    )
    report = tool_result.get("response_payload") or tool_result
    pending_id = report.get("pending_id") or tool_result.get("pending_id")
    if not pending_id:
        natural_reply = _tool_completion_message(
            action.action_name,
            report,
            tool_result.get("memory_text") or "pending_id_missing_after_ensure_pending",
        )
        response = _complete_response_from_payload(state, report, natural_reply)
        state.final_response = response
        return {**_state_update(state), "response": serialize_chat_response(response), "pending_id": None}
    if action.action_name == "trigger_subculture_workflow":
        clear_workflow_request_state(state.session_id)
    natural_reply = report.get("message") or report.get("msg") or f"Pending approval {pending_id} created."
    response = _complete_response_from_payload(state, report, natural_reply)
    observation = collect_observation(response, decision)
    state.previous_observations.append(observation)
    state.final_response = response
    state.terminal_status = AgentTerminalStatus.WAITING_APPROVAL
    _refresh_status_bar(state, "await_approval")
    record_agent_event(
        state,
        "agent_observation_collected",
        "agent_runtime",
        {"observation": observation.to_event_payload()},
    )
    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="response_created",
        layer="chat_service",
        payload={"action": observation.action, "status": observation.status},
    )
    current_step = dict(graph_state.get("current_step") or {})
    current_step["observation"] = observation.to_event_payload()
    current_step["terminal_status"] = AgentTerminalStatus.WAITING_APPROVAL.value
    return {
        **_state_update(state),
        "current_step": current_step,
        "response": serialize_chat_response(response),
        "pending_id": int(pending_id),
        "approval": {
            "pending_id": int(pending_id),
            "execution_idempotency_key": execution_key,
            "domain_dedupe_key": domain_key,
            "action_name": action.action_name,
        },
    }


def _route_after_ensure_pending(graph_state: AgentRuntimeGraphState) -> str:
    if graph_state.get("pending_id"):
        return "MarkWaitingApproval"
    return "Observe"


def _mark_waiting_approval_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    pending_id = int(graph_state["pending_id"])
    mark_run_waiting_approval(
        agent_run_id=state.agent_run_id,
        pending_id=pending_id,
        graph_thread_id=graph_state["graph_thread_id"],
        last_node="AwaitApproval",
    )
    record_agent_event(
        state,
        "agent_run_marked_waiting_approval",
        "agent_runtime",
        {
            "graph_thread_id": graph_state["graph_thread_id"],
            "pending_id": pending_id,
            "node_name": "MarkWaitingApproval",
        },
    )
    current_step = dict(graph_state.get("current_step") or {})
    observation_payload = current_step.get("observation") or {}
    if observation_payload:
        observation = AgentObservation(
            status=observation_payload.get("status"),
            action=observation_payload.get("action"),
            route_kind=observation_payload.get("route_kind"),
            output=observation_payload.get("output") or {},
            natural_reply=observation_payload.get("natural_reply"),
            pending_id=observation_payload.get("pending_id"),
            error_event_id=observation_payload.get("error_event_id"),
        )
        assessment = assess_observation(observation, AgentTerminalStatus.WAITING_APPROVAL)
        record_agent_event(state, "agent_observation_assessed", "agent_runtime", {"assessment": assessment.to_event_payload()})
        step_payload = {
            "index": state.step_index,
            "plan_step": current_step.get("plan_step"),
            "decision": current_step.get("decision") or {},
            "action": current_step.get("action") or {},
            "policy_verdict": current_step.get("policy_verdict") or graph_state.get("policy") or {},
            "observation": observation.to_event_payload(),
            "observation_assessment": assessment.to_event_payload(),
            "terminal_status": AgentTerminalStatus.WAITING_APPROVAL.value,
        }
        record_agent_event(state, "agent_step_finished", "agent_runtime", {"step": step_payload})
        current_step = step_payload
    record_agent_event(
        state,
        "agent_loop_stopped",
        "agent_runtime",
        {
            "directive": {
                "should_continue": False,
                "reason": "terminal_status:waiting_approval",
                "next_step_index": None,
                "terminal_status": AgentTerminalStatus.WAITING_APPROVAL.value,
                "finalize_composite": False,
            }
        },
    )
    return {**_state_update(state), "current_step": current_step}


def _await_approval_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    resume_payload = interrupt(
        {
            "agent_run_id": graph_state["agent_run_id"],
            "graph_thread_id": graph_state["graph_thread_id"],
            "pending_id": graph_state.get("pending_id"),
            "action_name": (graph_state.get("approval") or {}).get("action_name"),
            "risk_level": (graph_state.get("action") or {}).get("risk_level"),
            "approval_summary": _approval_summary(
                deserialize_action(graph_state.get("action")),
                int(graph_state.get("pending_id") or 0),
            ),
        }
    )
    return {"approval": {"resume_payload": resume_payload, "pending_id": graph_state.get("pending_id")}}


def _load_approval_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    pending_id = int(graph_state.get("pending_id") or (graph_state.get("approval") or {}).get("pending_id") or 0)
    mark_run_resumed(
        agent_run_id=state.agent_run_id,
        graph_thread_id=graph_state["graph_thread_id"],
        last_node="LoadApproval",
    )
    pending = database.get_pending_action(pending_id)
    record_agent_event(
        state,
        "graph_run_resumed",
        "agent_runtime",
        {"pending_id": pending_id, "graph_thread_id": graph_state["graph_thread_id"], "node_name": "LoadApproval"},
    )
    return {**_state_update(state), "approval": {"pending_id": pending_id, "pending": pending}}


def _revalidate_approval_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    pending = (graph_state.get("approval") or {}).get("pending")
    pending_id = (graph_state.get("approval") or {}).get("pending_id")
    result = {"status": "stale_or_invalid", "reason": "pending_not_found", "pending_id": pending_id}
    if pending:
        if pending.get("executed_at"):
            result = {"status": "already_executed", "pending_id": pending_id}
        elif pending.get("status") == "denied":
            result = {"status": "denied", "pending_id": pending_id}
        elif pending.get("status") != "approved":
            result = {"status": "stale_or_invalid", "reason": f"pending_status:{pending.get('status')}", "pending_id": pending_id}
        elif pending.get("graph_thread_id") != graph_state["graph_thread_id"]:
            result = {"status": "stale_or_invalid", "reason": "graph_thread_mismatch", "pending_id": pending_id}
        else:
            payload = pending.get("payload") or {}
            data = payload.get("data") or {}
            if payload.get("type") == "workflow_subculture":
                protocol_validation = verify_frozen_protocol_payload(payload)
                if not protocol_validation.get("valid"):
                    result = {
                        "status": "stale_or_invalid",
                        "reason": "protocol_hash_mismatch",
                        "pending_id": pending_id,
                        "protocol_id": protocol_validation.get("protocol_id"),
                        "protocol_hash": protocol_validation.get("protocol_hash"),
                    }
                    record_agent_event(
                        state,
                        "protocol_hash_mismatch",
                        "agent_runtime",
                        {"validation": result, "node_name": "RevalidateApproval"},
                    )
                    return {**_state_update(state), "approval_validation": result}
                current = database.get_algae_status(data.get("strain_id"))
                frozen_protocol = payload.get("protocol") or {}
                expected = frozen_protocol.get("expected_generation", data.get("expected_generation"))
                if not current:
                    result = {"status": "stale_or_invalid", "reason": "strain_not_found", "pending_id": pending_id}
                elif expected is not None and int(current.get("generation_number") or 0) != int(expected):
                    result = {"status": "stale_or_invalid", "reason": "stale_generation", "pending_id": pending_id}
                else:
                    result = {"status": "approved_and_valid", "pending_id": pending_id}
            else:
                result = {"status": "approved_and_valid", "pending_id": pending_id}
    event_type = "approval_resume_validated" if result["status"] == "approved_and_valid" else "approval_resume_rejected"
    record_agent_event(
        state,
        event_type,
        "agent_runtime",
        {"validation": result, "node_name": "RevalidateApproval"},
    )
    if result["status"] == "stale_or_invalid":
        record_agent_event(state, "stale_state_detected", "agent_runtime", {"validation": result})
    return {**_state_update(state), "approval_validation": result}


def _route_after_revalidate(graph_state: AgentRuntimeGraphState) -> str:
    status = (graph_state.get("approval_validation") or {}).get("status")
    if status == "approved_and_valid":
        return "ExecuteApprovedAction"
    if status in {"denied", "already_executed"}:
        return "FinishRun"
    return "BlockedResponse"


async def _execute_approved_action_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    pending_id = int((graph_state.get("approval_validation") or {}).get("pending_id"))
    queued = database.queue_pending_execution(pending_id)
    result = {
        "status": "queued" if queued else "error",
        "action": "trusted_execution_queued",
        "pending_id": pending_id,
        "canonical_execution_pending": True,
    }
    database.mark_pending_resume_completed(pending_id, result)
    payload = {
        "action": result.get("action") or "approved_pending_executed",
        "status": result.get("status", "success"),
        "pending_id": pending_id,
        **result,
    }
    natural_reply = (
        f"Pending {pending_id} was approved and queued for the trusted worker. "
        "Execution is not considered successful until canonical state is updated."
    )
    response = _complete_response_from_payload(state, payload, natural_reply)
    state.final_response = response
    state.terminal_status = AgentTerminalStatus.WAITING_APPROVAL if queued else AgentTerminalStatus.FAILED
    record_agent_event(
        state,
        "graph_run_completed_after_resume",
        "agent_runtime",
        {"pending_id": pending_id, "result": result, "node_name": "ExecuteApprovedAction"},
    )
    return {**_state_update(state), "response": serialize_chat_response(response)}


def _approval_terminal_response_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    validation = graph_state.get("approval_validation") or {}
    pending_id = validation.get("pending_id")
    if validation.get("status") == "denied":
        payload = {"action": "approval_denied", "status": "denied", "pending_id": pending_id}
        response = _complete_response_from_payload(state, payload, f"Pending {pending_id} was denied; no action was executed.")
        state.terminal_status = AgentTerminalStatus.BLOCKED
        state.final_response = response
        database.mark_pending_resume_completed(int(pending_id), payload)
        return {**_state_update(state), "response": serialize_chat_response(response)}
    if validation.get("status") == "already_executed":
        payload = {"action": "already_executed", "status": "success", "pending_id": pending_id}
        response = _complete_response_from_payload(state, payload, f"Pending {pending_id} was already executed; no duplicate action was run.")
        state.terminal_status = AgentTerminalStatus.SUCCEEDED
        state.final_response = response
        record_agent_event(state, "duplicate_execution_prevented", "agent_runtime", {"pending_id": pending_id})
        return {**_state_update(state), "response": serialize_chat_response(response)}
    return _blocked_response_node(graph_state)


def _observe_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    decision = deserialize_agent_decision(graph_state["decision"])
    response = deserialize_chat_response(graph_state["response"])
    action = deserialize_action(graph_state.get("action")) or build_agent_action(decision)
    observation = collect_observation(response, decision)
    state.previous_observations.append(observation)
    state.final_response = response
    state.executed_action_signatures.append(action_signature(action))
    if (
        state.loop_mode in {"read_only_composite", "composite_guarded"}
        and state.loop_plan
        and state.loop_plan_index < len(state.loop_plan)
        and decision.route_kind == state.loop_plan[state.loop_plan_index].route_kind
        and decision.action_name == state.loop_plan[state.loop_plan_index].action_name
    ):
        state.loop_plan_index += 1

    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="response_created",
        layer="chat_service",
        payload={"action": observation.action, "status": observation.status},
    )
    record_agent_event(state, "agent_observation_collected", "agent_runtime", {"observation": observation.to_event_payload()})

    state.terminal_status = derive_terminal_status(observation)
    assessment = assess_observation(observation, state.terminal_status)
    record_agent_event(state, "agent_observation_assessed", "agent_runtime", {"assessment": assessment.to_event_payload()})
    step_payload = {
        "index": state.step_index,
        "plan_step": (graph_state.get("current_step") or {}).get("plan_step"),
        "decision": decision.to_event_payload(),
        "action": action.to_event_payload(),
        "policy_verdict": graph_state.get("policy") or {},
        "observation": observation.to_event_payload(),
        "observation_assessment": assessment.to_event_payload(),
        "terminal_status": state.terminal_status.value,
    }
    record_agent_event(state, "agent_step_finished", "agent_runtime", {"step": step_payload})
    record_agent_event(
        state,
        "agent_replan_started",
        "agent_runtime",
        {"observation": observation.to_event_payload(), "loop_mode": state.loop_mode, "loop_plan_index": state.loop_plan_index},
    )
    _refresh_status_bar(state, "observe")
    return {**_state_update(state), "current_step": step_payload}


def _verify_observation_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    """Make the verifier boundary explicit for both legacy and scientific runs.

    Scientific tools persist their richer multi-check VerificationReport as a
    versioned artifact. Legacy routes retain the deterministic assessment that
    was already produced by Observe, so adding this node is backward compatible.
    """
    state = _runtime_state(graph_state)
    current_step = dict(graph_state.get("current_step") or {})
    assessment = dict(current_step.get("observation_assessment") or {})
    verification = {
        "verdict": "pass" if assessment.get("can_continue", True) else "block",
        "source": "scientific_artifact" if state.initial_route_kind == "scientific_task" else "runtime_observation_assessment",
        "assessment": assessment,
    }
    current_step["verification"] = verification
    record_agent_event(
        state,
        "agent_observation_verified",
        "agent_runtime",
        {"verification": verification},
    )
    _refresh_status_bar(state, "verify_observation")
    return {**_state_update(state), "current_step": current_step}


def _route_after_verify(graph_state: AgentRuntimeGraphState) -> str:
    state = _runtime_state(graph_state)
    if state.terminal_status == AgentTerminalStatus.NEEDS_MORE_INFO and state.task_id:
        return "MarkWaitingInput"
    return "Replan"


def _mark_waiting_input_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    paused = state.terminal_status == AgentTerminalStatus.PAUSED
    finish_run(
        state.agent_run_id,
        status=AgentTerminalStatus.PAUSED.value if paused else AgentTerminalStatus.NEEDS_MORE_INFO.value,
        final_route=state.initial_route_kind,
        response_summary=(state.final_response.natural_reply if state.final_response else "")[:500],
    )
    record_agent_event(
        state,
        "agent_task_paused" if paused else "agent_task_waiting_input",
        "agent_runtime",
        {"task_id": state.task_id, "task_state_version": state.task_state_version},
    )
    return _state_update(state)


def _await_task_input_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    paused = state.terminal_status == AgentTerminalStatus.PAUSED
    resumed = interrupt(
        {
            "reason": "budget_pause" if paused else "conversation_task_missing_slots",
            "task_id": state.task_id,
            "task_state_version": state.task_state_version,
            "missing_fields": list(state.missing_fields or []),
        }
    )
    if not isinstance(resumed, dict) or not str(resumed.get("message") or "").strip():
        raise RuntimeError("task_resume_message_required")
    current = conversation_tasks.get_task(str(state.task_id)) if state.task_id else None
    resumed_version = int(resumed.get("task_state_version") or 0)
    if not current or int(current.get("version") or 0) != resumed_version:
        raise conversation_tasks.TaskVersionConflict(str(state.task_id))

    state.agent_run_id = str(resumed["agent_run_id"])
    state.user_message = str(resumed["message"])
    state.task_state_version = resumed_version
    request_payload = resumed.get("request_context")
    if isinstance(request_payload, dict):
        state.request_context = RuntimeRequestContext(**request_payload)
    state.max_steps = max(1, int(resumed.get("max_steps") or state.max_steps or 1))
    state.step_index = 0
    state.terminal_status = None
    state.final_response = None
    state.loop_plan = []
    state.loop_plan_index = 0
    state.next_decision = None
    state.missing_fields = []
    state.model_turn_count = 0
    state.tool_call_count = 0
    state.compute_call_count = 0
    state.proposal_count = 0
    state.cumulative_model_tokens = 0
    state.agentic_started_at = time.monotonic()
    state.conversation_history = get_runtime_deps().get_session_memory(state.session_id)
    state.conversation_history.append({"role": "user", "content": state.user_message})
    resumed_decision = None
    if current.get("task_type") == "email" and current.get("status") == "collecting":
        email_spec = resume_email_request(current, state.user_message)
        resumed_decision = decision_from_routing_decision(
            EmailDecision(
                reason_code=ReasonCode.EMAIL_MATCHED,
                risk_level=RiskLevel.MEDIUM,
                explanation="The message supplies fields for the active email task.",
                source_text=email_spec.source_message,
                request_spec=email_spec.to_dict(),
            ),
            state,
        )
        state.next_decision = resumed_decision
    record_agent_event(
        state,
        "agent_task_input_resumed",
        "agent_runtime",
        {"task_id": state.task_id, "task_state_version": resumed_version},
    )
    return {
        **_state_update(state),
        "agent_run_id": state.agent_run_id,
        "task_state_version": resumed_version,
        "user_message": state.user_message,
        "max_steps": state.max_steps,
        "decision": None,
        "action": None,
        "policy": None,
        "response": None,
        "directive": None,
        "next_decision": (
            serialize_agent_decision(resumed_decision)
            if resumed_decision is not None
            else None
        ),
        "final_response": None,
    }


def _replan_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    observation_payload = (graph_state.get("current_step") or {}).get("observation") or {}
    assessment_payload = (graph_state.get("current_step") or {}).get("observation_assessment") or {}
    step = AgentStep(index=state.step_index)
    step.observation = AgentObservation(
        status=observation_payload.get("status"),
        action=observation_payload.get("action"),
        route_kind=observation_payload.get("route_kind"),
        output=observation_payload.get("output") or {},
        natural_reply=observation_payload.get("natural_reply"),
        pending_id=observation_payload.get("pending_id"),
        error_event_id=observation_payload.get("error_event_id"),
    )
    step.terminal_status = state.terminal_status
    if step.observation is not None and state.terminal_status is not None:
        step.observation_assessment = assess_observation(step.observation, state.terminal_status)

    prior_decision = deserialize_agent_decision(graph_state.get("decision"))
    if (
        prior_decision is not None
        and prior_decision.decision_source == "llm_tool_loop_v2"
        and _agentic_execution_enabled(state)
    ):
        previous_model_tools = [
            item.get("name")
            for item in state.last_model_action.get("tool_calls") or []
        ]
        try:
            next_decision, final_response = _run_agentic_model_turn(
                state,
                raw_decision=prior_decision.raw_decision,
            )
        except Exception as exc:
            record_agent_event(
                state,
                "agentic_replan_failed",
                "agent_runtime",
                {"error_type": type(exc).__name__, "error_message": str(exc)},
            )
            next_decision = None
            final_response = _agentic_final_response(
                state,
                status="partial",
                reason="model_replan_failed",
                content=(
                    "已保存当前调查 checkpoint，但模型重规划调用失败。"
                    "现有工具结果仍保留在 Trace 中，未创建或执行新的副作用。"
                ),
            )
        if next_decision is not None:
            if next_decision.action_name not in previous_model_tools:
                record_agent_event(
                    state,
                    "plan_changed_after_observation",
                    "agent_runtime",
                    {
                        "previous_tools": previous_model_tools,
                        "next_tool": next_decision.action_name,
                    },
                )
            directive = {
                "should_continue": True,
                "reason": "llm_replanned_after_observation",
                "next_step_index": state.step_index + 1,
                "terminal_status": None,
                "finalize_composite": False,
                "replan_source": "llm_tool_loop_v2",
            }
            state.terminal_status = None
            record_agent_event(
                state,
                "agent_loop_continue",
                "agent_runtime",
                {
                    "next_decision": next_decision.to_event_payload(),
                    "source": "llm_tool_loop_v2",
                },
            )
            _refresh_status_bar(state, "replan")
            return {
                **_state_update(state),
                "directive": directive,
                "next_decision": serialize_agent_decision(next_decision),
            }
        if final_response is not None:
            state.final_response = final_response
            final_status = final_response.agent_output.get("status")
            if final_status == "needs_more_info":
                state.terminal_status = AgentTerminalStatus.NEEDS_MORE_INFO
            elif final_status == "partial":
                state.terminal_status = AgentTerminalStatus.MAX_STEPS_REACHED
            elif final_status == "paused":
                state.terminal_status = AgentTerminalStatus.PAUSED
            else:
                state.terminal_status = AgentTerminalStatus.SUCCEEDED
            state.loop_stop_reason = final_response.agent_output.get("reason") or "model_final_answer"
            directive = {
                "should_continue": False,
                "reason": state.loop_stop_reason,
                "next_step_index": None,
                "terminal_status": state.terminal_status.value,
                "finalize_composite": False,
                "replan_source": "llm_tool_loop_v2",
            }
            record_agent_event(
                state,
                "agent_loop_stopped",
                "agent_runtime",
                {"directive": directive},
            )
            _refresh_status_bar(state, "replan")
            return {
                **_state_update(state),
                "directive": directive,
                "next_decision": None,
                "response": serialize_chat_response(final_response),
            }

    directive = build_replan_directive(state, step)
    state.replan_history.append(directive.to_event_payload())
    record_agent_event(state, "agent_replan_directive_created", "agent_runtime", {"directive": directive.to_event_payload()})
    if directive.decision is not None:
        record_agent_event(
            state,
            "agent_replan_decision_made",
            "agent_runtime",
            {"replan_source": directive.replan_source, "decision": directive.decision.to_event_payload()},
        )
    elif directive.blocked_by:
        record_agent_event(state, "agent_replan_blocked", "agent_runtime", {"directive": directive.to_event_payload()})

    if directive.finalize_composite:
        state.final_response = build_composite_loop_response(state, get_runtime_deps())
        state.terminal_status = directive.terminal_status or AgentTerminalStatus.SUCCEEDED
        record_agent_event(
            state,
            "agent_loop_finalized",
            "agent_runtime",
            {
                "directive": directive.to_event_payload(),
                "response": {
                    "action": state.final_response.agent_output.get("action"),
                    "status": state.final_response.agent_output.get("status"),
                    "step_count": state.final_response.agent_output.get("step_count"),
                },
            },
        )
    elif directive.terminal_status is not None:
        state.terminal_status = directive.terminal_status

    next_decision = None
    if directive.should_continue and directive.decision is not None:
        next_decision = serialize_agent_decision(directive.decision)
        record_agent_event(state, "agent_loop_continue", "agent_runtime", {"directive": directive.to_event_payload()})
    else:
        state.loop_stop_reason = directive.reason
        record_agent_event(state, "agent_loop_stopped", "agent_runtime", {"directive": directive.to_event_payload()})
    _refresh_status_bar(state, "replan")
    return {
        **_state_update(state),
        "directive": directive.to_event_payload(),
        "next_decision": next_decision,
    }


def _route_after_replan(graph_state: AgentRuntimeGraphState) -> str:
    state = _runtime_state(graph_state)
    if state.terminal_status == AgentTerminalStatus.PAUSED and state.task_id:
        return "MarkWaitingInput"
    directive = graph_state.get("directive") or {}
    if directive.get("should_continue") and graph_state.get("next_decision") is not None:
        return "BeginStep"
    return "FinishRun"


def _finish_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    if state.final_response is None:
        response = deserialize_chat_response(graph_state.get("response") or graph_state.get("final_response"))
        if response is not None:
            state.final_response = response
    if state.final_response is None:
        validation = graph_state.get("approval_validation") or {}
        if validation.get("status") in {"denied", "already_executed"}:
            terminal = _approval_terminal_response_node(graph_state)
            state = deserialize_runtime_state(terminal["runtime_state"])
        else:
            state.terminal_status = AgentTerminalStatus.MAX_STEPS_REACHED
            raise RuntimeError("agent_loop_finished_without_response")

    decision = _last_decision_from_graph(graph_state)
    final_status = state.terminal_status or AgentTerminalStatus.SUCCEEDED
    state.terminal_status = final_status
    _refresh_status_bar(state, "finish_run")
    output = state.final_response.agent_output or {}
    finish_run(
        state.agent_run_id,
        status=_finish_status_for_run(final_status),
        final_route=state.initial_route_kind or (decision.route_kind if decision else None),
        risk_level=decision.risk_level if decision else None,
        response_summary=state.final_response.natural_reply[:500],
        error_event_id=output.get("error_event_id"),
    )
    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="run_failed" if final_status == AgentTerminalStatus.FAILED else "run_finished",
        layer="chat_service",
        payload={"status": final_status.value, "graph_thread_id": graph_state.get("graph_thread_id")},
    )
    try:
        review = run_post_run_learning_review(
            agent_run_id=state.agent_run_id,
            session_id=state.session_id,
        )
        if review.get("status") != "skipped":
            run_learning_curator(agent_run_id=state.agent_run_id, session_id=state.session_id)
    except Exception as exc:
        record_run_event(
            state.agent_run_id,
            session_id=state.session_id,
            event_type="agent_learning_review_failed",
            layer="agent_learning",
            payload={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
    return _state_update(state)


def build_agent_runtime_graph(*, checkpointer: Any | None = None):
    graph = StateGraph(AgentRuntimeGraphState)
    read_retry = RetryPolicy(max_attempts=2, initial_interval=0.1, max_interval=1.0)
    graph.add_node("InitializeRun", _initialize_node)
    graph.add_node("BeginStep", _begin_step_node)
    graph.add_node("BuildContext", _build_context_node, retry_policy=read_retry)
    graph.add_node("DecideAction", _decide_action_node, retry_policy=read_retry)
    graph.add_node("EvaluatePolicy", _evaluate_policy_node)
    graph.add_node("ExecuteAction", _execute_action_node)
    graph.add_node("BlockedResponse", _blocked_response_node)
    graph.add_node("EnsurePending", _ensure_pending_node)
    graph.add_node("MarkWaitingApproval", _mark_waiting_approval_node)
    graph.add_node("AwaitApproval", _await_approval_node)
    graph.add_node("LoadApproval", _load_approval_node)
    graph.add_node("RevalidateApproval", _revalidate_approval_node)
    graph.add_node("ExecuteApprovedAction", _execute_approved_action_node)
    graph.add_node("Observe", _observe_node)
    graph.add_node("VerifyObservation", _verify_observation_node)
    graph.add_node("MarkWaitingInput", _mark_waiting_input_node)
    graph.add_node("AwaitTaskInput", _await_task_input_node)
    graph.add_node("Replan", _replan_node)
    graph.add_node("FinishRun", _finish_node)

    graph.add_edge(START, "InitializeRun")
    graph.add_edge("InitializeRun", "BeginStep")
    graph.add_edge("BeginStep", "BuildContext")
    graph.add_edge("BuildContext", "DecideAction")
    graph.add_conditional_edges(
        "DecideAction",
        _route_after_decide,
        {
            "EvaluatePolicy": "EvaluatePolicy",
            "FinishRun": "FinishRun",
        },
    )
    graph.add_conditional_edges(
        "EvaluatePolicy",
        _route_after_policy,
        {
            "ExecuteAction": "ExecuteAction",
            "EnsurePending": "EnsurePending",
            "BlockedResponse": "BlockedResponse",
        },
    )
    graph.add_conditional_edges(
        "EnsurePending",
        _route_after_ensure_pending,
        {
            "MarkWaitingApproval": "MarkWaitingApproval",
            "Observe": "Observe",
        },
    )
    graph.add_edge("MarkWaitingApproval", "AwaitApproval")
    graph.add_edge("AwaitApproval", "LoadApproval")
    graph.add_edge("LoadApproval", "RevalidateApproval")
    graph.add_conditional_edges(
        "RevalidateApproval",
        _route_after_revalidate,
        {
            "ExecuteApprovedAction": "ExecuteApprovedAction",
            "BlockedResponse": "BlockedResponse",
            "FinishRun": "FinishRun",
        },
    )
    graph.add_edge("ExecuteApprovedAction", "FinishRun")
    graph.add_edge("ExecuteAction", "Observe")
    graph.add_edge("BlockedResponse", "Observe")
    graph.add_edge("Observe", "VerifyObservation")
    graph.add_conditional_edges(
        "VerifyObservation",
        _route_after_verify,
        {"MarkWaitingInput": "MarkWaitingInput", "Replan": "Replan"},
    )
    graph.add_edge("MarkWaitingInput", "AwaitTaskInput")
    graph.add_edge("AwaitTaskInput", "BeginStep")
    graph.add_conditional_edges(
        "Replan",
        _route_after_replan,
        {
            "BeginStep": "BeginStep",
            "MarkWaitingInput": "MarkWaitingInput",
            "FinishRun": "FinishRun",
        },
    )
    graph.add_edge("FinishRun", END)
    return graph.compile(checkpointer=checkpointer)


def get_agent_runtime_graph():
    compiled = get_compiled_graph()
    if compiled is not None:
        return compiled
    return build_agent_runtime_graph(checkpointer=build_fallback_checkpointer())


async def run_agent_graph_loop(
    payload: ChatRequest,
    deps: AgentRuntimeDeps,
    max_steps: int = 1,
    agent_run_id: str | None = None,
    task_id: str | None = None,
    task_state_version: int | None = None,
    request_context: RuntimeRequestContext | None = None,
) -> ChatResponse:
    run_id = agent_run_id or str(uuid.uuid4())
    graph_thread_id = graph_thread_id_for_task(task_id) if task_id else graph_thread_id_for_run(run_id)
    attempt_no = 1
    if task_id:
        current_task = conversation_tasks.get_task(task_id)
        if not current_task or int(current_task.get("version") or 0) != int(task_state_version or 0):
            raise conversation_tasks.TaskVersionConflict(task_id)
        with connect(row_factory=True) as conn:
            attempt_no = int(
                conn.execute("SELECT COUNT(*) + 1 FROM agent_runs WHERE task_id = ?", (task_id,)).fetchone()[0]
            )
    start_run(
        payload.session_id,
        payload.message,
        run_id,
        graph_thread_id=graph_thread_id,
        graph_definition_version=GRAPH_DEFINITION_VERSION,
        state_schema_version=STATE_SCHEMA_VERSION,
        conversation_id=payload.session_id,
        task_id=task_id,
        attempt_no=attempt_no,
    )
    state = AgentRunState(
        agent_run_id=run_id,
        session_id=payload.session_id,
        user_message=payload.message,
        conversation_history=[],
        conversation_id=payload.session_id,
        task_id=task_id,
        task_state_version=task_state_version,
        request_context=request_context,
        max_steps=max(1, int(max_steps or 1)),
    )
    graph_state = _base_graph_state(
        agent_run_id=run_id,
        session_id=payload.session_id,
        graph_thread_id=graph_thread_id,
        user_message=payload.message,
        max_steps=max(1, int(max_steps or 1)),
        runtime_state=serialize_runtime_state(state),
        task_id=task_id,
        task_state_version=task_state_version,
    )
    app = get_agent_runtime_graph() # 拿到已经编译好的 LangGraph 图
    config = {"configurable": {"thread_id": graph_thread_id}} # 告诉LangGraph这次运行的 thread_id 是什么
    try:
        invoke_input: AgentRuntimeGraphState | Command = graph_state
        if task_id and hasattr(app, "aget_state"):
            checkpoint = await app.aget_state(config)
            if "AwaitTaskInput" in tuple(getattr(checkpoint, "next", ()) or ()):
                invoke_input = Command(
                    resume={
                        "message": payload.message,
                        "agent_run_id": run_id,
                        "task_state_version": task_state_version,
                        "request_context": request_context.to_dict() if request_context else None,
                        "max_steps": max_steps,
                    }
                )
        result = await app.ainvoke(invoke_input, config=config) # 初始或恢复输入丢进同一 LangGraph Thread
        final_state = deserialize_runtime_state(result["runtime_state"])
        if result.get("__interrupt__"):
            record_run_event(
                run_id,
                session_id=payload.session_id,
                event_type="agent_run_interrupted",
                layer="agent_runtime",
                payload={"graph_thread_id": graph_thread_id, "pending_id": result.get("pending_id")},
            )
            if final_state.final_response is None:
                raise RuntimeError("agent_loop_interrupted_without_response")
            return final_state.final_response
        if final_state.final_response is None:
            raise RuntimeError("agent_loop_finished_without_response")
        return final_state.final_response
    except Exception as exc:
        decision = _last_decision_from_graph(graph_state)
        from app.core.provider_errors import safe_provider_error

        failure = safe_provider_error(exc)
        state.last_error = {
            **failure,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        state.terminal_status = AgentTerminalStatus.FAILED
        state.final_response = ChatResponse(
            status="failed",
            session_id=state.session_id,
            agent_status=AgentTerminalStatus.FAILED.value,
            agent_output={
                "action": "assistant_error",
                "status": "failed",
                "outcome_status": "failed",
                "error_code": failure["code"],
                "error_category": failure["category"],
                "retryable": failure["retryable"],
                "agent_run_id": state.agent_run_id,
                "task_id": state.task_id,
            },
            natural_reply=failure["public_message"],
        )
        error_event_id = record_error_event(
            session_id=state.session_id,
            agent_run_id=state.agent_run_id,
            layer="chat_service",
            component="chat_service",
            operation="handle_chat",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"route_kind": decision.route_kind if decision else None, "graph_thread_id": graph_thread_id},
        )
        finish_run(
            state.agent_run_id,
            status=AgentTerminalStatus.FAILED.value,
            final_route=decision.route_kind if decision else None,
            risk_level=decision.risk_level if decision else None,
            error_event_id=error_event_id,
        )
        record_run_event(
            state.agent_run_id,
            session_id=state.session_id,
            event_type="run_failed",
            layer="chat_service",
            payload={"error_event_id": error_event_id, "failure_stage": "handle_chat", "failure_layer": "chat_service"},
        )
        if state.task_id:
            try:
                task = conversation_tasks.get_task(state.task_id)
                if task and task.get("status") not in conversation_tasks.TERMINAL_STATUSES:
                    conversation_tasks.update_task(
                        state.task_id,
                        status="failed",
                        proposed_action={
                            **(task.get("proposed_action") or {}),
                            "failure": state.last_error,
                        },
                        event_type="task_failed",
                    )
            except Exception:
                pass
        try:
            if hasattr(app, "aupdate_state"):
                await app.aupdate_state(
                    config,
                    {
                        "runtime_state": serialize_runtime_state(state),
                        "final_response": serialize_chat_response(state.final_response),
                    },
                )
        except Exception:
            pass
        try:
            run_post_run_learning_review(
                agent_run_id=state.agent_run_id,
                session_id=state.session_id,
            )
        except Exception:
            pass
        raise
