from __future__ import annotations

from app.services.agent_runtime.decision import decision_from_routing_decision
from app.services.agent_runtime.executor import build_agent_action
from app.services.agent_runtime.loop_control import action_signature
from app.services.agent_runtime.state import (
    AgentDecision,
    AgentObservation,
    AgentRunState,
    AgentStep,
    AgentTerminalStatus,
    ObservationAssessment,
    ReplanCandidate,
    ReplanResult,
)
from app.services.agent_runtime.llm_replan_provider import (
    LlmReplanSuggestion,
    collect_llm_replan_suggestions,
)
from app.services.intent.routing_models import (
    ClarificationDecision,
    EntityRef,
    ReasonCode,
    RiskLevel,
    WorkflowDecision,
)


TERMINAL_STOP_STATUSES = {
    AgentTerminalStatus.WAITING_APPROVAL,
    AgentTerminalStatus.NEEDS_MORE_INFO,
    AgentTerminalStatus.FAILED,
    AgentTerminalStatus.BLOCKED,
}


def _is_empty_result(observation: AgentObservation) -> bool:
    output = observation.output or {}
    if observation.route_kind == "lab_query" and output.get("action") == "query_due_subculture":
        return not output.get("strains")
    return False


def _assessment_payload(assessment: ObservationAssessment | None) -> dict:
    return assessment.to_event_payload() if assessment else {}


def assess_observation(
    observation: AgentObservation,
    terminal_status: AgentTerminalStatus,
) -> ObservationAssessment:
    if terminal_status in TERMINAL_STOP_STATUSES:
        return ObservationAssessment(
            status=observation.status,
            can_continue=False,
            should_replan=False,
            reason=f"terminal_status:{terminal_status.value}",
            terminal_status=terminal_status,
            category=terminal_status.value,
            deterministic_reason=f"terminal_status:{terminal_status.value}",
        )
    if observation.error_event_id is not None:
        return ObservationAssessment(
            status=observation.status,
            can_continue=False,
            should_replan=False,
            reason="observation_has_error_event",
            terminal_status=AgentTerminalStatus.FAILED,
            category="tool_error",
            deterministic_reason="observation_has_error_event",
        )
    category = "empty_result" if _is_empty_result(observation) else "success"
    return ObservationAssessment(
        status=observation.status,
        can_continue=True,
        should_replan=True,
        reason="observation_success",
        terminal_status=terminal_status,
        category=category,
        deterministic_reason="observation_success",
    )


def _replan_result(
    *,
    should_continue: bool,
    reason: str,
    decision: AgentDecision | None = None,
    next_step_index: int | None = None,
    terminal_status: AgentTerminalStatus | None = None,
    finalize_composite: bool = False,
    is_dynamic: bool = False,
    blocked_by: str | None = None,
    replan_source: str = "none",
    deterministic_assessment: dict | None = None,
    llm_assessment: dict | None = None,
    rule_candidates: list[ReplanCandidate] | None = None,
    llm_candidates: list[ReplanCandidate] | None = None,
    selected_candidate: ReplanCandidate | None = None,
    rejected_candidates: list[ReplanCandidate] | None = None,
    selection_reason: str | None = None,
    policy_boundary: str | None = None,
) -> ReplanResult:
    return ReplanResult(
        should_continue=should_continue,
        reason=reason,
        decision=decision,
        next_step_index=next_step_index,
        terminal_status=terminal_status,
        finalize_composite=finalize_composite,
        is_dynamic=is_dynamic,
        blocked_by=blocked_by,
        replan_source=replan_source,
        deterministic_assessment=deterministic_assessment or {},
        llm_assessment=llm_assessment or {},
        rule_candidates=rule_candidates or [],
        llm_candidates=llm_candidates or [],
        selected_candidate=selected_candidate,
        rejected_candidates=rejected_candidates or [],
        selection_reason=selection_reason,
        policy_boundary=policy_boundary,
    )


def _is_plan_mode(state: AgentRunState) -> bool:
    return state.loop_mode in {"read_only_composite", "composite_guarded", "dynamic_replan"}


def _next_static_plan_decision(state: AgentRunState) -> AgentDecision | None:
    if not state.loop_plan or not _is_plan_mode(state):
        return None
    if state.loop_plan_index >= len(state.loop_plan):
        return None
    return state.loop_plan[state.loop_plan_index]


def _goal_mentions_subculture_execution(goal: str | None) -> bool:
    text = (goal or "").casefold()
    execution_terms = (
        "execute",
        "run",
        "start",
        "trigger",
        "workflow",
        "subculture",
        "\u6267\u884c",
    )
    return any(term in text for term in execution_terms)


def _due_subculture_rows(observation: AgentObservation) -> list[dict]:
    output = observation.output or {}
    if observation.route_kind != "lab_query":
        return []
    if output.get("action") != "query_due_subculture":
        return []
    return [item for item in output.get("strains") or [] if isinstance(item, dict)]


def _known_strain_ids(state: AgentRunState, observation: AgentObservation) -> set[str]:
    ids = {
        str(item.get("strain_id"))
        for item in state.current_db_snapshot.get("strains", [])
        if isinstance(item, dict) and item.get("strain_id")
    }
    ids.update(
        str(item.get("strain_id"))
        for item in _due_subculture_rows(observation)
        if item.get("strain_id")
    )
    return ids


def _existing_workflow_pending_for_target(state: AgentRunState, target_id: str | None) -> dict | None:
    if not target_id:
        return None
    for item in state.pending_actions or []:
        if item.get("status") != "pending":
            continue
        if "workflow" not in str(item.get("action_type") or ""):
            continue
        if str(item.get("target") or "") == str(target_id):
            return item
        payload = item.get("payload") or {}
        data = payload.get("data") or {}
        if str(data.get("strain_id") or "") == str(target_id):
            return item
    return None


def _rule_replan_candidates(state: AgentRunState, observation: AgentObservation) -> list[ReplanCandidate]:
    if not _goal_mentions_subculture_execution(state.user_goal):
        return []
    rows = _due_subculture_rows(observation)
    if not rows:
        if observation.route_kind == "lab_query" and observation.output.get("action") == "query_due_subculture":
            return [
                ReplanCandidate(
                    action_type="stop_with_reason",
                    source="rule",
                    reason="No due subculture targets were returned by the lab query observation.",
                    confidence=1.0,
                    risk_level="low",
                    metadata={"observation_action": observation.action},
                )
            ]
        return []
    if len(rows) > 1:
        ids = [str(item.get("strain_id")) for item in rows if item.get("strain_id")]
        return [
            ReplanCandidate(
                action_type="ask_clarification",
                source="rule",
                reason="Multiple due subculture targets were observed; high-risk workflow requests must stay single-target.",
                confidence=1.0,
                route_kind="clarification",
                question=(
                    "Multiple strains are due for subculture. Please choose one strain_id before creating a workflow approval request: "
                    + ", ".join(ids)
                ),
                risk_level="high",
                metadata={"target_ids": ids},
            )
        ]
    strain_id = rows[0].get("strain_id")
    if not strain_id:
        return [
            ReplanCandidate(
                action_type="stop_with_reason",
                source="rule",
                reason="The due subculture observation did not include a strain_id.",
                confidence=1.0,
                risk_level="high",
                blocked_by="invalid_observation",
            )
        ]
    existing = _existing_workflow_pending_for_target(state, str(strain_id))
    if existing:
        return [
            ReplanCandidate(
                action_type="explain_result",
                source="rule",
                reason="An existing workflow pending request already covers the observed target.",
                confidence=1.0,
                target_id=str(strain_id),
                explanation=(
                    f"{strain_id} already has a pending workflow approval request "
                    f"(pending {existing.get('id')}); no duplicate pending request will be created."
                ),
                risk_level="high",
                metadata={"pending_id": existing.get("id"), "policy_boundary": "existing_workflow_pending"},
            )
        ]
    return [
        ReplanCandidate(
            action_type="propose_workflow_approval",
            source="rule",
            reason="A single due subculture target was returned and the user goal asks to run the workflow.",
            confidence=1.0,
            route_kind="workflow",
            target_id=str(strain_id),
            risk_level="high",
        )
    ]


def _llm_replan_candidates(
    state: AgentRunState,
    observation: AgentObservation,
    suggestion: LlmReplanSuggestion,
) -> list[ReplanCandidate]:
    known_ids = _known_strain_ids(state, observation)
    candidates: list[ReplanCandidate] = []
    for item in suggestion.candidates:
        target_id = item.target_ids[0] if len(item.target_ids) == 1 else None
        blocked_by = None
        if item.action_type == "propose_workflow_approval":
            if observation.route_kind == "knowledge_query":
                blocked_by = "rag_boundary"
            elif len(item.target_ids) > 1:
                blocked_by = "multiple_targets"
            elif item.target_ids and target_id not in known_ids:
                blocked_by = "target_not_in_context"
        candidates.append(
            ReplanCandidate(
                action_type=item.action_type,
                source="llm",
                reason=item.reason,
                confidence=item.confidence,
                route_kind="workflow" if item.action_type == "propose_workflow_approval" else None,
                target_id=target_id,
                question=item.question,
                explanation=item.explanation,
                risk_level="high" if item.action_type == "propose_workflow_approval" else "low",
                blocked_by=blocked_by,
                metadata={
                    "missing_fields": list(item.missing_fields),
                    "risk_notes": list(item.risk_notes),
                    "target_ids": list(item.target_ids),
                },
            )
        )
    return candidates


def _workflow_decision_for_target_id(state: AgentRunState, strain_id: str | None) -> AgentDecision | None:
    if not strain_id:
        return None
    raw_decision = WorkflowDecision(
        reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
        risk_level=RiskLevel.HIGH,
        explanation="Dynamic replanning selected a single due strain for workflow approval.",
        source_text=state.user_message,
        target=EntityRef("strain", strain_id, strain_id, "database_id", True),
    )
    decision = decision_from_routing_decision(raw_decision, state)
    decision.decision_source = "dynamic_replan"
    return decision


def _clarification_decision_from_candidate(
    state: AgentRunState,
    candidate: ReplanCandidate,
    *,
    response_action: str,
    default_question: str,
) -> AgentDecision:
    raw_decision = ClarificationDecision(
        reason_code=ReasonCode.REQUIRED_FIELDS_MISSING,
        risk_level=RiskLevel.HIGH if candidate.risk_level == "high" else RiskLevel.LOW,
        explanation=candidate.reason,
        source_text=state.user_message,
        questions=(candidate.question or candidate.explanation or default_question,),
        response_action=response_action,
    )
    decision = decision_from_routing_decision(raw_decision, state)
    decision.decision_source = "dynamic_replan"
    return decision


def _decision_from_replan_candidate(
    state: AgentRunState,
    candidate: ReplanCandidate,
) -> AgentDecision | None:
    if candidate.action_type == "propose_workflow_approval":
        return _workflow_decision_for_target_id(state, candidate.target_id)
    if candidate.action_type == "ask_clarification":
        return _clarification_decision_from_candidate(
            state,
            candidate,
            response_action="dynamic_replan_clarification",
            default_question="Please clarify the target or the next step before I create any approval request.",
        )
    if candidate.action_type == "explain_result":
        return _clarification_decision_from_candidate(
            state,
            candidate,
            response_action="dynamic_replan_explain_result",
            default_question="I can explain the observation, but I will not take further action without a safe next step.",
        )
    if candidate.action_type == "stop_with_reason":
        return _clarification_decision_from_candidate(
            state,
            candidate,
            response_action="dynamic_replan_stop_reason",
            default_question="No safe follow-up action was selected from the observation.",
        )
    return None


def _select_replan_candidate(
    *,
    state: AgentRunState,
    observation: AgentObservation,
    rule_candidates: list[ReplanCandidate],
    llm_candidates: list[ReplanCandidate],
) -> tuple[ReplanCandidate | None, list[ReplanCandidate], str, str | None]:
    candidates = rule_candidates + llm_candidates
    rejected = [item for item in candidates if item.blocked_by]
    usable = [item for item in candidates if not item.blocked_by]

    if observation.route_kind == "knowledge_query":
        for item in usable:
            if item.action_type in {"ask_clarification", "explain_result", "stop_with_reason"}:
                return item, rejected, "rag_observation_limited_to_explain_or_clarify", None
        return None, rejected, "rag_observation_not_authoritative_for_actions", "rag_boundary"

    for item in usable:
        if item.source == "rule" and item.action_type == "ask_clarification":
            return item, rejected, "rule_clarification_for_multiple_or_missing_targets", None

    for item in usable:
        if item.source == "rule" and item.action_type == "propose_workflow_approval":
            return item, rejected, "rule_single_target_workflow_candidate", None

    workflow_candidates = [
        item
        for item in usable
        if item.action_type == "propose_workflow_approval"
    ]
    if workflow_candidates:
        best = max(workflow_candidates, key=lambda item: item.confidence)
        if not best.target_id:
            clarification = ReplanCandidate(
                action_type="ask_clarification",
                source="system",
                reason="LLM proposed workflow approval but did not provide a unique verified target.",
                confidence=1.0,
                question="Please provide one strain_id before I create any workflow approval request.",
                risk_level="high",
                blocked_by=None,
                metadata={"rejected_action_type": best.action_type},
            )
            rejected.append(best)
            return clarification, rejected, "llm_workflow_missing_unique_target", None
        existing = _existing_workflow_pending_for_target(state, best.target_id)
        if existing:
            explanation = ReplanCandidate(
                action_type="explain_result",
                source="system",
                reason="Existing pending workflow approval already covers the target.",
                confidence=1.0,
                target_id=best.target_id,
                explanation=(
                    f"{best.target_id} already has pending workflow approval "
                    f"(pending {existing.get('id')}); no duplicate pending request will be created."
                ),
                risk_level="high",
                metadata={"pending_id": existing.get("id"), "policy_boundary": "existing_workflow_pending"},
            )
            rejected.append(best)
            return explanation, rejected, "existing_pending_boundary", "existing_workflow_pending"
        return best, rejected, "llm_workflow_candidate_selected_after_deterministic_checks", None

    for action_type in ("ask_clarification", "explain_result", "stop_with_reason"):
        compatible = [item for item in usable if item.action_type == action_type]
        if compatible:
            return max(compatible, key=lambda item: item.confidence), rejected, f"{action_type}_candidate_selected", None

    return None, rejected, "no_safe_replan_candidate", None


def _dynamic_decision_from_observation(
    state: AgentRunState,
    step: AgentStep,
) -> ReplanResult:
    observation = step.observation
    deterministic_assessment = _assessment_payload(step.observation_assessment)
    if observation is None:
        return _replan_result(
            should_continue=False,
            reason="missing_observation",
            terminal_status=step.terminal_status,
            blocked_by="missing_observation",
            deterministic_assessment=deterministic_assessment,
        )

    if state.dynamic_plan_count >= state.max_dynamic_replans:
        return _replan_result(
            should_continue=False,
            reason="max_dynamic_replans_reached",
            terminal_status=step.terminal_status,
            blocked_by="max_dynamic_replans",
            deterministic_assessment=deterministic_assessment,
        )

    suggestion = collect_llm_replan_suggestions(state, observation)
    llm_assessment = (
        suggestion.assessment.to_event_payload()
        if suggestion.assessment
        else {}
    )
    rule_candidates = _rule_replan_candidates(state, observation)
    llm_candidates = _llm_replan_candidates(state, observation, suggestion)
    selected, rejected, selection_reason, policy_boundary = _select_replan_candidate(
        state=state,
        observation=observation,
        rule_candidates=rule_candidates,
        llm_candidates=llm_candidates,
    )

    if selected is None:
        return _replan_result(
            should_continue=False,
            reason=selection_reason if rule_candidates or llm_candidates else "no_loop_plan",
            terminal_status=step.terminal_status,
            blocked_by=policy_boundary,
            deterministic_assessment=deterministic_assessment,
            llm_assessment=llm_assessment,
            rule_candidates=rule_candidates,
            llm_candidates=llm_candidates,
            rejected_candidates=rejected,
            selection_reason=selection_reason,
            policy_boundary=policy_boundary,
        )

    if selected.source == "rule" and selected.action_type == "stop_with_reason":
        return _replan_result(
            should_continue=False,
            reason="no_due_subculture_targets",
            terminal_status=step.terminal_status,
            deterministic_assessment=deterministic_assessment,
            llm_assessment=llm_assessment,
            rule_candidates=rule_candidates,
            llm_candidates=llm_candidates,
            selected_candidate=selected,
            rejected_candidates=rejected,
            selection_reason=selection_reason,
        )

    decision = _decision_from_replan_candidate(state, selected)
    if decision is None:
        return _replan_result(
            should_continue=False,
            reason="selected_replan_candidate_not_dispatchable",
            terminal_status=step.terminal_status,
            blocked_by="not_dispatchable",
            deterministic_assessment=deterministic_assessment,
            llm_assessment=llm_assessment,
            rule_candidates=rule_candidates,
            llm_candidates=llm_candidates,
            selected_candidate=selected,
            rejected_candidates=rejected,
            selection_reason=selection_reason,
            policy_boundary=policy_boundary,
        )

    next_action = build_agent_action(decision)
    if action_signature(next_action) in state.executed_action_signatures:
        return _replan_result(
            should_continue=False,
            reason="duplicate_action",
            terminal_status=step.terminal_status,
            finalize_composite=state.loop_mode in {"read_only_composite", "composite_guarded"},
            blocked_by="duplicate_action",
            deterministic_assessment=deterministic_assessment,
            llm_assessment=llm_assessment,
            rule_candidates=rule_candidates,
            llm_candidates=llm_candidates,
            selected_candidate=selected,
            rejected_candidates=rejected,
            selection_reason=selection_reason,
            policy_boundary=policy_boundary,
        )

    state.dynamic_plan_count += 1
    state.loop_mode = "dynamic_replan"
    return _replan_result(
        should_continue=True,
        reason="dynamic_observation_replan",
        decision=decision,
        next_step_index=state.step_index + 1,
        is_dynamic=True,
        replan_source="dynamic_observation",
        deterministic_assessment=deterministic_assessment,
        llm_assessment=llm_assessment,
        rule_candidates=rule_candidates,
        llm_candidates=llm_candidates,
        selected_candidate=selected,
        rejected_candidates=rejected,
        selection_reason=selection_reason,
        policy_boundary=policy_boundary,
    )


def plan_next_action_from_observation(
    state: AgentRunState,
    step: AgentStep,
) -> ReplanResult:
    assessment = step.observation_assessment
    if assessment is None:
        return _replan_result(
            should_continue=False,
            reason="missing_observation_assessment",
            terminal_status=step.terminal_status,
            blocked_by="missing_observation_assessment",
        )

    if not assessment.can_continue:
        return _replan_result(
            should_continue=False,
            reason=assessment.reason,
            terminal_status=assessment.terminal_status or step.terminal_status,
            blocked_by="terminal_status",
        )

    if (
        state.loop_plan
        and state.loop_mode in {"read_only_composite", "composite_guarded"}
        and state.loop_plan_index >= len(state.loop_plan)
    ):
        return _replan_result(
            should_continue=False,
            reason="loop_plan_complete",
            terminal_status=AgentTerminalStatus.SUCCEEDED,
            finalize_composite=True,
            replan_source="static_plan",
        )

    if state.step_index >= state.max_steps:
        return _replan_result(
            should_continue=False,
            reason="max_steps_reached",
            terminal_status=AgentTerminalStatus.MAX_STEPS_REACHED,
            blocked_by="max_steps",
        )

    static_decision = _next_static_plan_decision(state)
    if static_decision is not None:
        next_action = build_agent_action(static_decision)
        if action_signature(next_action) in state.executed_action_signatures:
            return _replan_result(
                should_continue=False,
                reason="duplicate_action",
                terminal_status=step.terminal_status,
                finalize_composite=state.loop_mode in {"read_only_composite", "composite_guarded"},
                blocked_by="duplicate_action",
                replan_source="static_plan",
            )

        return _replan_result(
            should_continue=True,
            reason="next_static_plan_step",
            decision=static_decision,
            next_step_index=state.loop_plan_index + 1,
            replan_source="static_plan",
        )

    return _dynamic_decision_from_observation(state, step)


def build_replan_directive(
    state: AgentRunState,
    step: AgentStep,
) -> ReplanResult:
    return plan_next_action_from_observation(state, step)
