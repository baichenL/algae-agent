from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any

from app.schemas.algae import ChatResponse
from app.services.agent_runtime.state import (
    AgentAction,
    AgentDecision,
    AgentObservation,
    AgentPlan,
    AgentPlanStep,
    AgentPolicyVerdict,
    AgentRunState,
    RuntimeRequestContext,
    AgentStep,
    AgentTerminalStatus,
)
from app.services.intent import routing_models as rm
from app.services.context.status_bar import build_status_bar


STATE_SCHEMA_VERSION = 3
GRAPH_DEFINITION_VERSION = "agent-runtime-graph-v2"
DECISION_SCHEMA_VERSION = 3
ACTION_SCHEMA_VERSION = 2

LEGACY_GRAPH_VERSIONS = {
    (
        1,
        "agent-runtime-graph-v1",
        1,
        1,
    ),
    (
        2,
        "agent-runtime-graph-v2",
        2,
        2,
    ),
}


def _safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _safe(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_safe(item) for item in value]
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, SimpleNamespace):
        return {str(key): _safe(item) for key, item in vars(value).items()}
    if hasattr(value, "__dict__") and not callable(value):
        return {str(key): _safe(item) for key, item in vars(value).items()}
    return value


def serialize_chat_response(response: ChatResponse | None) -> dict[str, Any] | None:
    if response is None:
        return None
    return response.model_dump()


def deserialize_chat_response(payload: dict[str, Any] | None) -> ChatResponse | None:
    if payload is None:
        return None
    return ChatResponse(**payload)


def serialize_routing_decision(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    payload = _safe(decision)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    payload["_decision_type"] = type(decision).__name__
    return payload


def _enum(enum_cls, value):
    if value is None or isinstance(value, enum_cls):
        return value
    if isinstance(value, dict) and set(value) == {"value"}:
        value = value["value"]
    try:
        return enum_cls(value)
    except (TypeError, ValueError):
        # Unknown future/hostile enum values must survive checkpoint recovery so
        # the policy layer can reject them; deserialization is not authorization.
        return value


def _entity(payload: dict[str, Any] | None):
    if not payload:
        return None
    return rm.EntityRef(
        entity_type=payload.get("entity_type"),
        canonical_id=payload.get("canonical_id"),
        mention=payload.get("mention") or "",
        source=payload.get("source"),
        exists=bool(payload.get("exists")),
    )


def _candidate(payload: dict[str, Any]) -> rm.RouteCandidate:
    return rm.RouteCandidate(
        kind=_enum(rm.RouteKind, payload.get("kind")),
        reason_code=_enum(rm.ReasonCode, payload.get("reason_code")),
        risk_level=_enum(rm.RiskLevel, payload.get("risk_level")),
        evidence=payload.get("evidence") or "",
        source=payload.get("source") or "rule",
        score=float(payload.get("score") or 1.0),
        evidence_items=tuple(payload.get("evidence_items") or ()),
        negative_signals=tuple(payload.get("negative_signals") or ()),
        selection_reason=payload.get("selection_reason"),
        speech_act=_enum(rm.SpeechAct, payload.get("speech_act")) if payload.get("speech_act") else None,
        target=_entity(payload.get("target")),
        entity_options=tuple(_entity(item) for item in payload.get("entity_options") or () if item),
        query_type=payload.get("query_type"),
        operation=payload.get("operation"),
        tool_name=payload.get("tool_name"),
        arguments=payload.get("arguments") or {},
        missing_fields=tuple(payload.get("missing_fields") or ()),
        form_candidates=tuple(payload.get("form_candidates") or ()),
    )


def deserialize_routing_decision(payload: dict[str, Any] | None) -> Any:
    if payload is None:
        return None
    decision_type = payload.get("_decision_type")
    if decision_type in {"SimpleNamespace", "namespace"}:
        data = {key: value for key, value in payload.items() if key != "_decision_type"}
        data["target"] = _entity(data.get("target"))
        data["entity_options"] = tuple(
            _entity(item)
            for item in data.get("entity_options") or ()
            if isinstance(item, dict)
        )
        data["candidates"] = tuple(
            _candidate(item)
            if isinstance(item, dict) and item.get("kind") is not None
            else SimpleNamespace(**item)
            if isinstance(item, dict)
            else item
            for item in data.get("candidates") or ()
        )
        data["steps"] = tuple(
            deserialize_routing_decision(item)
            if isinstance(item, dict) and item.get("_decision_type")
            else item
            for item in data.get("steps") or ()
        )
        return SimpleNamespace(**data)
    common = {
        "reason_code": _enum(rm.ReasonCode, payload.get("reason_code")),
        "risk_level": _enum(rm.RiskLevel, payload.get("risk_level")),
        "explanation": payload.get("explanation") or "",
        "source_text": payload.get("source_text") or "",
        "target": _entity(payload.get("target")),
        "candidates": tuple(_candidate(item) for item in payload.get("candidates") or ()),
        "selection_trace": payload.get("selection_trace") or {},
    }
    if decision_type == "ToolInfoDecision":
        return rm.ToolInfoDecision(**common)
    if decision_type == "EmailDecision":
        return rm.EmailDecision(
            **common,
            request_spec=payload.get("request_spec") or {},
        )
    if decision_type == "ForbiddenDecision":
        return rm.ForbiddenDecision(
            **common,
            capability_request=payload.get("capability_request") or {},
        )
    if decision_type == "QueryDecision":
        return rm.QueryDecision(
            **common,
            kind=_enum(rm.RouteKind, payload.get("kind")),
            query_type=payload.get("query_type") or "strain_status",
            targets=tuple(_entity(item) for item in payload.get("targets") or () if item),
        )
    if decision_type == "PendingFormDecision":
        return rm.PendingFormDecision(
            **common,
            form_action=payload.get("form_action") or "start",
            operation=payload.get("operation"),
            tool_name=payload.get("tool_name"),
            arguments=payload.get("arguments") or {},
            missing_fields=tuple(payload.get("missing_fields") or ()),
            form_candidates=tuple(payload.get("form_candidates") or ()),
        )
    if decision_type == "KnowledgeDecision":
        return rm.KnowledgeDecision(**common)
    if decision_type == "WriteDecision":
        return rm.WriteDecision(
            **common,
            operation=payload.get("operation") or "update",
            tool_name=payload.get("tool_name") or "update_algae_strain",
            arguments=payload.get("arguments") or {},
            missing_fields=tuple(payload.get("missing_fields") or ()),
        )
    if decision_type == "WorkflowDecision":
        return rm.WorkflowDecision(
            **common,
            speech_act=_enum(rm.SpeechAct, payload.get("speech_act")) or rm.SpeechAct.COMMAND,
            workflow_name=payload.get("workflow_name") or "subculture",
        )
    if decision_type == "WorkflowAuditDecision":
        return rm.WorkflowAuditDecision(
            **common,
            speech_act=_enum(rm.SpeechAct, payload.get("speech_act")) or rm.SpeechAct.DIAGNOSTIC,
            pending_id=payload.get("pending_id"),
            workflow_name=payload.get("workflow_name") or "subculture",
        )
    if decision_type == "ScientificTaskDecision":
        return rm.ScientificTaskDecision(
            **common,
            arguments=payload.get("arguments") or {},
            missing_fields=tuple(payload.get("missing_fields") or ()),
        )
    if decision_type == "ChatDecision":
        return rm.ChatDecision(**common)
    if decision_type == "ClarificationDecision":
        return rm.ClarificationDecision(
            **common,
            questions=tuple(payload.get("questions") or ()),
            response_action=payload.get("response_action") or "routing_clarification",
        )
    if decision_type == "CompositeDecision":
        return rm.CompositeDecision(
            **common,
            steps=tuple(deserialize_routing_decision(item) for item in payload.get("steps") or ()),
        )
    return SimpleNamespace(**{key: value for key, value in payload.items() if key != "_decision_type"})


def serialize_agent_decision(decision: AgentDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    payload = decision.to_event_payload()
    payload["raw_decision"] = serialize_routing_decision(decision.raw_decision)
    return payload


def deserialize_agent_decision(payload: dict[str, Any] | None) -> AgentDecision | None:
    if payload is None:
        return None
    raw = deserialize_routing_decision(payload.get("raw_decision"))
    return AgentDecision(
        route_kind=payload.get("route_kind") or "unknown",
        action_type=payload.get("action_type") or "unknown",
        action_name=payload.get("action_name") or "unknown",
        action_args=payload.get("action_args") or {},
        risk_level=payload.get("risk_level") or "none",
        requires_approval=bool(payload.get("requires_approval")),
        missing_fields=list(payload.get("missing_fields") or []),
        can_continue=bool(payload.get("can_continue")),
        reason=payload.get("reason") or "",
        confidence=payload.get("confidence"),
        target=payload.get("target"),
        candidate_routes=list(payload.get("candidate_routes") or []),
        decision_source=payload.get("decision_source") or "intent_router",
        terminal_hint=payload.get("terminal_hint"),
        raw_decision=raw,
    )


def serialize_runtime_state(state: AgentRunState) -> dict[str, Any]:
    return {
        "agent_run_id": state.agent_run_id,
        "session_id": state.session_id,
        "user_message": state.user_message,
        "conversation_history": state.conversation_history,
        "conversation_id": state.conversation_id,
        "task_id": state.task_id,
        "task_state_version": state.task_state_version,
        "request_context": state.request_context.to_dict() if state.request_context else None,
        "user_goal": state.user_goal,
        "max_steps": state.max_steps,
        "step_index": state.step_index,
        "runtime_context": state.runtime_context,
        "context_history": state.context_history,
        "status_bar": state.status_bar,
        "compression_records": state.compression_records,
        "selected_skill_definitions": state.selected_skill_definitions,
        "skill_selection_trace": state.skill_selection_trace,
        "model_input_reports": state.model_input_reports,
        "context_budget_report": state.context_budget_report,
        "previous_observations": [item.to_event_payload() for item in state.previous_observations],
        "current_db_snapshot": state.current_db_snapshot,
        "pending_actions": state.pending_actions,
        "session_memory": state.session_memory,
        "user_memories": state.user_memories,
        "rag_evidence": state.rag_evidence,
        "workflow_status": state.workflow_status,
        "missing_fields": state.missing_fields,
        "policy_history": state.policy_history,
        "agent_plan": state.agent_plan.to_event_payload() if state.agent_plan else None,
        "loop_plan": [serialize_agent_decision(item) for item in state.loop_plan],
        "loop_plan_index": state.loop_plan_index,
        "loop_mode": state.loop_mode,
        "loop_stop_reason": state.loop_stop_reason,
        "executed_action_signatures": state.executed_action_signatures,
        "initial_route_kind": state.initial_route_kind,
        "loop_plan_event_recorded": state.loop_plan_event_recorded,
        "replan_history": state.replan_history,
        "dynamic_plan_count": state.dynamic_plan_count,
        "max_dynamic_replans": state.max_dynamic_replans,
        "planning_context": state.planning_context,
        "agentic_state_schema_version": state.agentic_state_schema_version,
        "agentic_mode": state.agentic_mode,
        "safety_envelope": state.safety_envelope,
        "resolved_tools": state.resolved_tools,
        "v2_observations": state.v2_observations,
        "hypotheses": state.hypotheses,
        "candidate_plans": state.candidate_plans,
        "model_turn_count": state.model_turn_count,
        "tool_call_count": state.tool_call_count,
        "compute_call_count": state.compute_call_count,
        "proposal_count": state.proposal_count,
        "plan_patch_count": state.plan_patch_count,
        "cumulative_model_tokens": state.cumulative_model_tokens,
        "agentic_started_at": state.agentic_started_at,
        "last_model_action": state.last_model_action,
        "terminal_status": state.terminal_status.value if state.terminal_status else None,
        "final_response": serialize_chat_response(state.final_response),
        "last_error": state.last_error,
    }


def _plan_from_payload(payload: dict[str, Any] | None) -> AgentPlan | None:
    if not payload:
        return None
    return AgentPlan(
        mode=payload.get("mode") or "single_step",
        reason=payload.get("reason") or "",
        steps=[
            AgentPlanStep(
                index=int(item.get("index") or 0),
                route_kind=item.get("route_kind") or "",
                action_name=item.get("action_name") or "",
                action_type=item.get("action_type") or "",
                risk_level=item.get("risk_level") or "none",
                purpose=item.get("purpose") or "",
                expected_effect=item.get("expected_effect") or "",
            )
            for item in payload.get("steps") or []
        ],
    )


def deserialize_runtime_state(payload: dict[str, Any]) -> AgentRunState:
    state = AgentRunState(
        agent_run_id=payload["agent_run_id"],
        session_id=payload["session_id"],
        user_message=payload["user_message"],
        conversation_history=list(payload.get("conversation_history") or []),
        conversation_id=payload.get("conversation_id") or payload.get("session_id"),
        task_id=payload.get("task_id"),
        task_state_version=payload.get("task_state_version"),
        request_context=(
            RuntimeRequestContext(**payload["request_context"])
            if payload.get("request_context")
            else None
        ),
        user_goal=payload.get("user_goal"),
        max_steps=int(payload.get("max_steps") or 1),
    )
    state.step_index = int(payload.get("step_index") or 0)
    state.runtime_context = payload.get("runtime_context") or {}
    state.context_history = list(payload.get("context_history") or [])
    state.status_bar = dict(payload.get("status_bar") or {})
    state.compression_records = list(payload.get("compression_records") or [])
    state.selected_skill_definitions = list(payload.get("selected_skill_definitions") or [])
    state.skill_selection_trace = list(payload.get("skill_selection_trace") or [])
    state.model_input_reports = list(payload.get("model_input_reports") or [])
    state.context_budget_report = dict(payload.get("context_budget_report") or {})
    state.previous_observations = [
        AgentObservation(
            status=item.get("status"),
            action=item.get("action"),
            route_kind=item.get("route_kind"),
            output=item.get("output") or {},
            natural_reply=item.get("natural_reply"),
            pending_id=item.get("pending_id"),
            error_event_id=item.get("error_event_id"),
        )
        for item in payload.get("previous_observations") or []
    ]
    state.current_db_snapshot = payload.get("current_db_snapshot") or {}
    state.pending_actions = payload.get("pending_actions") or []
    state.session_memory = payload.get("session_memory") or {}
    state.user_memories = list(payload.get("user_memories") or [])
    state.rag_evidence = payload.get("rag_evidence") or []
    state.workflow_status = payload.get("workflow_status") or {}
    state.missing_fields = list(payload.get("missing_fields") or [])
    state.policy_history = list(payload.get("policy_history") or [])
    state.agent_plan = _plan_from_payload(payload.get("agent_plan"))
    state.loop_plan = [item for item in (deserialize_agent_decision(item) for item in payload.get("loop_plan") or []) if item]
    state.loop_plan_index = int(payload.get("loop_plan_index") or 0)
    state.loop_mode = payload.get("loop_mode")
    state.loop_stop_reason = payload.get("loop_stop_reason")
    state.executed_action_signatures = list(payload.get("executed_action_signatures") or [])
    state.initial_route_kind = payload.get("initial_route_kind")
    state.loop_plan_event_recorded = bool(payload.get("loop_plan_event_recorded"))
    state.replan_history = list(payload.get("replan_history") or [])
    state.dynamic_plan_count = int(payload.get("dynamic_plan_count") or 0)
    state.max_dynamic_replans = int(payload.get("max_dynamic_replans") or 2)
    state.planning_context = payload.get("planning_context") or {}
    state.agentic_state_schema_version = int(payload.get("agentic_state_schema_version") or 2)
    state.agentic_mode = payload.get("agentic_mode")
    state.safety_envelope = dict(payload.get("safety_envelope") or {})
    state.resolved_tools = list(payload.get("resolved_tools") or [])
    state.v2_observations = list(payload.get("v2_observations") or [])
    state.hypotheses = list(payload.get("hypotheses") or [])
    state.candidate_plans = list(payload.get("candidate_plans") or [])
    state.model_turn_count = int(payload.get("model_turn_count") or 0)
    state.tool_call_count = int(payload.get("tool_call_count") or 0)
    state.compute_call_count = int(payload.get("compute_call_count") or 0)
    state.proposal_count = int(payload.get("proposal_count") or 0)
    state.plan_patch_count = int(payload.get("plan_patch_count") or 0)
    state.cumulative_model_tokens = int(payload.get("cumulative_model_tokens") or 0)
    state.agentic_started_at = payload.get("agentic_started_at")
    state.last_model_action = dict(payload.get("last_model_action") or {})
    state.terminal_status = AgentTerminalStatus(payload["terminal_status"]) if payload.get("terminal_status") else None
    state.final_response = deserialize_chat_response(payload.get("final_response"))
    state.last_error = payload.get("last_error")
    if not state.status_bar:
        state.status_bar = build_status_bar(state).to_dict()
        state.runtime_context["status_bar"] = state.status_bar
    return state


def serialize_action(action: AgentAction | None) -> dict[str, Any] | None:
    return action.to_event_payload() if action else None


def deserialize_action(payload: dict[str, Any] | None) -> AgentAction | None:
    if not payload:
        return None
    return AgentAction(
        action_type=payload.get("action_type") or "unknown",
        action_name=payload.get("action_name") or "unknown",
        action_args=payload.get("action_args") or {},
        risk_level=payload.get("risk_level") or "none",
        requires_approval=bool(payload.get("requires_approval")),
    )


def serialize_policy(verdict: AgentPolicyVerdict | None) -> dict[str, Any] | None:
    return verdict.to_event_payload() if verdict else None


def deserialize_policy(payload: dict[str, Any] | None) -> AgentPolicyVerdict | None:
    if not payload:
        return None
    return AgentPolicyVerdict(
        allowed=bool(payload.get("allowed")),
        category=payload.get("category") or "unknown",
        reason=payload.get("reason") or "",
        risk_level=payload.get("risk_level") or "none",
        requires_approval=bool(payload.get("requires_approval")),
        forbidden=bool(payload.get("forbidden")),
        policy_name=payload.get("policy_name") or "agent_runtime_policy_v1",
        evidence=payload.get("evidence") or {},
    )


def validate_graph_versions(graph_state: dict[str, Any]) -> bool:
    versions = (
        graph_state.get("state_schema_version"),
        graph_state.get("graph_definition_version"),
        graph_state.get("decision_schema_version"),
        graph_state.get("action_schema_version"),
    )
    return versions == (
        STATE_SCHEMA_VERSION,
        GRAPH_DEFINITION_VERSION,
        DECISION_SCHEMA_VERSION,
        ACTION_SCHEMA_VERSION,
    ) or versions in LEGACY_GRAPH_VERSIONS
