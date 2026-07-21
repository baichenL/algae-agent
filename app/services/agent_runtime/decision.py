from __future__ import annotations

from typing import Any

from app.services.agent_runtime.policy import requires_approval_action
from app.services.agent_runtime.policy import is_forbidden_direct_action
from app.services.agent_runtime.events import record_agent_event
from app.services.agent_runtime.state import (
    AgentDecision,
    AgentPlan,
    AgentPlanStep,
    AgentRunState,
    AgentRuntimeDeps,
)
from app.services.intent.routing_models import RouteKind, RoutingInput


def _enum_value(value: Any) -> str:
    return getattr(value, "value", str(value))


def _route_kind(raw_decision: Any) -> str:
    return _enum_value(getattr(raw_decision, "kind", "unknown"))


def _decision_args(raw_decision: Any) -> dict[str, Any]:
    args = dict(getattr(raw_decision, "arguments", {}) or {})
    if getattr(raw_decision, "query_type", None):
        args["query_type"] = raw_decision.query_type
    if getattr(raw_decision, "workflow_name", None):
        args["workflow_name"] = raw_decision.workflow_name
    if getattr(raw_decision, "pending_id", None) is not None:
        args["pending_id"] = raw_decision.pending_id
    if getattr(raw_decision, "form_action", None):
        args["form_action"] = raw_decision.form_action
    return args


def _target_payload(raw_decision: Any) -> dict[str, Any] | None:
    target = getattr(raw_decision, "target", None)
    if target is None:
        return None
    return {
        "entity_type": getattr(target, "entity_type", None),
        "canonical_id": getattr(target, "canonical_id", None),
        "mention": getattr(target, "mention", None),
        "source": getattr(target, "source", None),
        "exists": getattr(target, "exists", None),
    }


def _candidate_routes(raw_decision: Any) -> list[str]:
    return [
        _enum_value(getattr(candidate, "kind", "unknown"))
        for candidate in (getattr(raw_decision, "candidates", ()) or ())
    ]


def infer_action_type(raw_decision: Any) -> str:
    route_kind = _route_kind(raw_decision)
    if route_kind in {
        RouteKind.TOOL_INFO.value,
        RouteKind.LAB_QUERY.value,
        RouteKind.PENDING_QUERY.value,
        RouteKind.KNOWLEDGE_QUERY.value,
        RouteKind.WORKFLOW_AUDIT.value,
        RouteKind.SCIENTIFIC_TASK.value,
    }:
        return "read"
    if route_kind in {RouteKind.WRITE_ACTION.value, RouteKind.WORKFLOW.value}:
        return "approval_request"
    if route_kind == RouteKind.EMAIL.value:
        return "draft_or_approval_request"
    if route_kind == RouteKind.PENDING_FORM.value:
        return "pending_form"
    if route_kind == RouteKind.CLARIFICATION.value:
        return "clarification"
    if route_kind == RouteKind.CHAT.value:
        return "chat"
    if route_kind == RouteKind.COMPOSITE.value:
        return "composite"
    return "unknown"


def infer_action_name(raw_decision: Any) -> str:
    explicit_tool = getattr(raw_decision, "tool_name", None)
    if explicit_tool:
        return explicit_tool

    route_kind = _route_kind(raw_decision)
    if route_kind == RouteKind.TOOL_INFO.value:
        return "tool_info"
    if route_kind == RouteKind.PENDING_QUERY.value:
        return "list_pending_actions"
    if route_kind == RouteKind.LAB_QUERY.value:
        return getattr(raw_decision, "query_type", None) or "list_strains"
    if route_kind == RouteKind.KNOWLEDGE_QUERY.value:
        return "retrieve_rag"
    if route_kind == RouteKind.SCIENTIFIC_TASK.value:
        return "growth_diagnosis_start"
    if route_kind == RouteKind.WORKFLOW.value:
        return "trigger_subculture_workflow"
    if route_kind == RouteKind.WORKFLOW_AUDIT.value:
        return "read_workflow_runs"
    if route_kind == RouteKind.EMAIL.value:
        return "email_draft"
    if route_kind == RouteKind.PENDING_FORM.value:
        return "pending_form"
    if route_kind == RouteKind.CLARIFICATION.value:
        return getattr(raw_decision, "response_action", None) or "clarification"
    if route_kind == RouteKind.CHAT.value:
        return "chat"
    if route_kind == RouteKind.COMPOSITE.value:
        return "composite"
    return route_kind


def infer_can_continue(raw_decision: Any, missing_fields: list[str]) -> bool:
    route_kind = _route_kind(raw_decision)
    if missing_fields:
        return False
    if route_kind in {RouteKind.CLARIFICATION.value, "unknown"}:
        return False
    return infer_action_type(raw_decision) != "unknown"


def infer_confidence(raw_decision: Any) -> float:
    route_kind = _route_kind(raw_decision)
    reason_code = _enum_value(getattr(raw_decision, "reason_code", ""))
    if route_kind == "unknown" or reason_code == "unknown_route":
        return 0.0
    if route_kind == RouteKind.CLARIFICATION.value:
        return 0.5
    if route_kind == RouteKind.CHAT.value:
        return 0.6
    return 1.0


def _terminal_hint(raw_decision: Any, missing_fields: list[str]) -> str | None:
    route_kind = _route_kind(raw_decision)
    if missing_fields:
        return "needs_more_info"
    if route_kind == RouteKind.CLARIFICATION.value:
        return "needs_clarification"
    if infer_action_type(raw_decision) == "unknown":
        return "blocked"
    return None


def decision_from_routing_decision(raw_decision: Any, state: AgentRunState) -> AgentDecision:
    route_kind = _route_kind(raw_decision)
    action_type = infer_action_type(raw_decision)
    action_name = infer_action_name(raw_decision)
    risk_level = _enum_value(getattr(raw_decision, "risk_level", "none"))
    missing_fields = list(getattr(raw_decision, "missing_fields", ()) or ())
    return AgentDecision(
        route_kind=route_kind,
        action_type=action_type,
        action_name=action_name,
        action_args=_decision_args(raw_decision),
        risk_level=risk_level,
        requires_approval=requires_approval_action(action_name, risk_level),
        missing_fields=missing_fields,
        can_continue=infer_can_continue(raw_decision, missing_fields),
        reason=getattr(raw_decision, "explanation", "") or _enum_value(getattr(raw_decision, "reason_code", "")),
        confidence=infer_confidence(raw_decision),
        target=_target_payload(raw_decision),
        candidate_routes=_candidate_routes(raw_decision),
        decision_source="intent_router",
        terminal_hint=_terminal_hint(raw_decision, missing_fields),
        raw_decision=raw_decision,
    )


GUARDED_COMPOSITE_ACTION_TYPES = {
    "read",
    "draft_or_approval_request",
    "approval_request",
}


def _is_guarded_composite_step(decision: AgentDecision) -> bool:
    return (
        decision.action_type in GUARDED_COMPOSITE_ACTION_TYPES
        and decision.can_continue
        and not is_forbidden_direct_action(decision.action_name)
    )


def _plan_step(index: int, decision: AgentDecision) -> AgentPlanStep:
    return AgentPlanStep(
        index=index,
        route_kind=decision.route_kind,
        action_name=decision.action_name,
        action_type=decision.action_type,
        risk_level=decision.risk_level,
        purpose=decision.reason,
        expected_effect="no_side_effect" if decision.action_type == "read" else decision.action_type,
    )


def _record_plan(state: AgentRunState, plan: AgentPlan) -> None:
    if state.loop_plan_event_recorded:
        return
    state.agent_plan = plan
    state.loop_plan_event_recorded = True
    record_agent_event(
        state,
        "agent_plan_created",
        "agent_runtime",
        {"plan": plan.to_event_payload()},
    )


def _build_guarded_composite_plan(raw_decision: Any, state: AgentRunState) -> list[AgentDecision]:
    if _route_kind(raw_decision) != RouteKind.COMPOSITE.value:
        return []

    steps = list(getattr(raw_decision, "steps", ()) or ())
    if not steps:
        return []

    plan: list[AgentDecision] = []
    for raw_step in steps:
        child_decision = decision_from_routing_decision(raw_step, state)
        child_decision.decision_source = "composite_plan"
        if not _is_guarded_composite_step(child_decision):
            return []
        plan.append(child_decision)
    return plan


def decide_next_action(state: AgentRunState, deps: AgentRuntimeDeps) -> AgentDecision:
    if state.loop_plan and state.loop_plan_index < len(state.loop_plan):
        return state.loop_plan[state.loop_plan_index]

    raw_decision = deps.build_routing_decision(
        RoutingInput(
            message=deps.normalize_input(state.user_message),
            session_id=state.session_id,
            context_snapshot=state.context_snapshot,
            active_pending_form=deps.get_pending_form_state(state.session_id),
            active_workflow_request=deps.get_workflow_request_state(state.session_id),
        )
    )
    decision = decision_from_routing_decision(raw_decision, state)
    state.initial_route_kind = state.initial_route_kind or decision.route_kind

    loop_plan = _build_guarded_composite_plan(raw_decision, state)
    if loop_plan:
        state.loop_plan = loop_plan
        state.loop_plan_index = 0
        state.loop_mode = "composite_guarded"
        _record_plan(
            state,
            AgentPlan(
                mode="composite_guarded",
                steps=[_plan_step(index, item) for index, item in enumerate(loop_plan, start=1)],
                reason="Composite route contains guarded child steps that must pass policy one by one.",
            ),
        )
        record_agent_event(
            state,
            "agent_loop_plan_created",
            "agent_runtime",
            {"step_count": len(loop_plan), "route_kinds": [item.route_kind for item in loop_plan]},
        )
        return state.loop_plan[state.loop_plan_index]

    _record_plan(
        state,
        AgentPlan(
            mode="single_step",
            steps=[_plan_step(1, decision)],
            reason="Router produced a single dispatchable decision.",
        ),
    )
    return decision
