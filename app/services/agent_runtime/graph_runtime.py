from __future__ import annotations

import hashlib
import json
import uuid
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
from app.services.agent_runtime.decision import decide_next_action
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
from app.services.agent_runtime.policy import evaluate_policy
from app.services.agent_runtime.replanning import assess_observation, build_replan_directive
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
from app.services.chat.response_builder import _complete_chat_response, _tool_completion_message, defer_response_persistence
from app.services.chat.workflow_request_state import clear_workflow_request_state
from app.services.observability.error_events import record_error_event
from app.services.protocols.pending_payload import verify_frozen_protocol_payload
from app.services.strains import strain_service
from app.services.learning.curator import run_learning_curator
from app.services.learning.reviewer import run_post_run_learning_review
from app.services.context.status_bar import build_status_bar


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
    target = getattr(raw_decision, "target", None)
    return getattr(target, "canonical_id", None) if target else None


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    target = getattr(candidate, "target", None)
    return {
        "route_kind": _safe_value(getattr(candidate, "kind", None)),
        "reason_code": _safe_value(getattr(candidate, "reason_code", None)),
        "risk_level": _safe_value(getattr(candidate, "risk_level", None)),
        "source": getattr(candidate, "source", "rule"),
        "score": getattr(candidate, "score", None),
        "evidence_items": list(getattr(candidate, "evidence_items", ()) or ()),
        "negative_signals": list(getattr(candidate, "negative_signals", ()) or ()),
        "selection_reason": getattr(candidate, "selection_reason", None),
        "target": getattr(target, "canonical_id", None) if target else None,
    }


def _log_routing_decision(state: AgentRunState, raw_decision: Any, deps: AgentRuntimeDeps) -> None:
    deps.append_decision_event(
        {
            "event": "chat_intent_routed",
            "session_id": state.session_id,
            "agent_run_id": state.agent_run_id,
            "route_kind": _safe_value(raw_decision.kind),
            "reason_code": _safe_value(raw_decision.reason_code),
            "risk_level": _safe_value(raw_decision.risk_level),
            "candidate_routes": [_safe_value(item.kind) for item in raw_decision.candidates],
            "requires_clarification": _safe_value(raw_decision.kind) == "clarification",
            "strain_id": _decision_target_id(raw_decision),
            "strain_ids": [
                item.canonical_id
                for candidate in raw_decision.candidates
                for item in candidate.entity_options
                if item.canonical_id
            ],
            "plan_steps": [_safe_value(step.kind) for step in getattr(raw_decision, "steps", ())],
            "selection_trace": getattr(raw_decision, "selection_trace", {}) or {},
            "candidate_evidence": [_candidate_payload(item) for item in raw_decision.candidates],
        }
    )


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
        decision = decide_next_action(state, deps)
    else:
        decision = next_decision

    raw_decision = decision.raw_decision
    if decision.route_kind == "scientific_task":
        # Scientific runs contain their own governed observation/verification
        # cycle and may resume after approval. Preserve the small ordinary-chat
        # budget while allowing the declared scientific ceiling.
        state.max_steps = max(state.max_steps, 24)
    _build_targeted_context(state, raw_decision, deps)
    _log_routing_decision(state, raw_decision, deps)
    record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type="routed",
        layer="intent_router",
        payload={
            "route_kind": decision.route_kind,
            "reason_code": _safe_value(raw_decision.reason_code),
            "risk_level": decision.risk_level,
            "candidate_routes": [_safe_value(item.kind) for item in raw_decision.candidates],
            "selection_trace": getattr(raw_decision, "selection_trace", {}) or {},
            "candidate_evidence": [_candidate_payload(item) for item in raw_decision.candidates],
        },
    )
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
    result = await strain_service.execute_approved_pending_action(pending_id)
    database.mark_pending_resume_completed(pending_id, result)
    payload = {
        "action": result.get("action") or "approved_pending_executed",
        "status": result.get("status", "success"),
        "pending_id": pending_id,
        **result,
    }
    natural_reply = result.get("msg") or result.get("message") or f"Pending {pending_id} review completed."
    response = _complete_response_from_payload(state, payload, natural_reply)
    state.final_response = response
    state.terminal_status = AgentTerminalStatus.SUCCEEDED if result.get("status") == "success" else AgentTerminalStatus.FAILED
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
    finish_run(
        state.agent_run_id,
        status=AgentTerminalStatus.NEEDS_MORE_INFO.value,
        final_route=state.initial_route_kind,
        response_summary=(state.final_response.natural_reply if state.final_response else "")[:500],
    )
    record_agent_event(
        state,
        "agent_task_waiting_input",
        "agent_runtime",
        {"task_id": state.task_id, "task_state_version": state.task_state_version},
    )
    return _state_update(state)


def _await_task_input_node(graph_state: AgentRuntimeGraphState) -> dict[str, Any]:
    state = _runtime_state(graph_state)
    resumed = interrupt(
        {
            "reason": "conversation_task_missing_slots",
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
    state.conversation_history = get_runtime_deps().get_session_memory(state.session_id)
    state.conversation_history.append({"role": "user", "content": state.user_message})
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
        "next_decision": None,
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
    graph.add_edge("DecideAction", "EvaluatePolicy")
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
        {"BeginStep": "BeginStep", "FinishRun": "FinishRun"},
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
        state.last_error = {"error_type": type(exc).__name__, "error_message": str(exc)}
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
        try:
            run_post_run_learning_review(
                agent_run_id=state.agent_run_id,
                session_id=state.session_id,
            )
        except Exception:
            pass
        raise
