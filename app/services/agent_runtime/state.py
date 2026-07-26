from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from app.schemas.algae import ChatResponse


class AgentTerminalStatus(str, Enum):
    SUCCEEDED = "succeeded"
    WAITING_APPROVAL = "waiting_approval"
    NEEDS_MORE_INFO = "needs_more_info"
    FAILED = "failed"
    BLOCKED = "blocked"
    MAX_STEPS_REACHED = "max_steps_reached"
    PAUSED = "paused"


@dataclass(frozen=True)
class ContextSlice:
    name: str
    source: str
    authority: str
    summary: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "authority": self.authority,
            "summary": self.summary,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class ContextBundle:
    fact_context: ContextSlice
    session_context: ContextSlice
    rag_context: ContextSlice
    tool_context: ContextSlice
    policy_context: ContextSlice
    user_memory_context: ContextSlice = field(default_factory=lambda: ContextSlice(
        name="user_memory_context",
        source="user_memories",
        authority="user_preference_advisory",
        summary={"memory_count": 0},
        payload={"memories": []},
    ))

    def to_event_payload(self) -> dict[str, Any]:
        memory_trace = self.user_memory_context.to_event_payload()
        memories = list((self.user_memory_context.payload or {}).get("memories") or [])
        memory_trace["payload"] = {
            "memories": [
                {key: item.get(key) for key in ("id", "predicate", "revision", "score")}
                for item in memories if isinstance(item, dict)
            ]
        }
        return {
            "fact_context": self.fact_context.to_event_payload(),
            "session_context": self.session_context.to_event_payload(),
            "user_memory_context": memory_trace,
            "rag_context": self.rag_context.to_event_payload(),
            "tool_context": self.tool_context.to_event_payload(),
            "policy_context": self.policy_context.to_event_payload(),
        }

    def to_model_payload(self) -> dict[str, Any]:
        return {
            "fact_context": self.fact_context.to_event_payload(),
            "session_context": self.session_context.to_event_payload(),
            "user_memory_context": self.user_memory_context.to_event_payload(),
            "rag_context": self.rag_context.to_event_payload(),
            "tool_context": self.tool_context.to_event_payload(),
            "policy_context": self.policy_context.to_event_payload(),
        }

    def to_model_view(self, allowed_slices: tuple[str, ...] | list[str]) -> dict[str, Any]:
        payload = self.to_model_payload()
        return {name: payload[name] for name in allowed_slices if name in payload}


@dataclass(frozen=True)
class AgentPlanStep:
    index: int
    route_kind: str
    action_name: str
    action_type: str
    risk_level: str
    purpose: str
    expected_effect: str

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "route_kind": self.route_kind,
            "action_name": self.action_name,
            "action_type": self.action_type,
            "risk_level": self.risk_level,
            "purpose": self.purpose,
            "expected_effect": self.expected_effect,
        }


@dataclass(frozen=True)
class AgentPlan:
    mode: str
    steps: list[AgentPlanStep]
    reason: str

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "step_count": len(self.steps),
            "steps": [step.to_event_payload() for step in self.steps],
        }


@dataclass(frozen=True)
class ReplanCandidate:
    action_type: str
    source: str
    reason: str
    confidence: float = 1.0
    route_kind: str | None = None
    target_id: str | None = None
    question: str | None = None
    explanation: str | None = None
    risk_level: str = "none"
    blocked_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "source": self.source,
            "reason": self.reason,
            "confidence": self.confidence,
            "route_kind": self.route_kind,
            "target_id": self.target_id,
            "question": self.question,
            "explanation": self.explanation,
            "risk_level": self.risk_level,
            "blocked_by": self.blocked_by,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ObservationAssessment:
    status: str
    can_continue: bool
    should_replan: bool
    reason: str
    terminal_status: AgentTerminalStatus | None = None
    category: str = "success"
    deterministic_reason: str | None = None
    llm_assessment: dict[str, Any] = field(default_factory=dict)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "can_continue": self.can_continue,
            "should_replan": self.should_replan,
            "reason": self.reason,
            "terminal_status": self.terminal_status.value if self.terminal_status else None,
            "category": self.category,
            "deterministic_reason": self.deterministic_reason or self.reason,
            "llm_assessment": self.llm_assessment,
        }


@dataclass(frozen=True)
class ReplanDirective:
    should_continue: bool
    reason: str
    next_step_index: int | None = None
    terminal_status: AgentTerminalStatus | None = None
    finalize_composite: bool = False

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "should_continue": self.should_continue,
            "reason": self.reason,
            "next_step_index": self.next_step_index,
            "terminal_status": self.terminal_status.value if self.terminal_status else None,
            "finalize_composite": self.finalize_composite,
        }


@dataclass(frozen=True)
class ReplanResult:
    should_continue: bool
    reason: str
    decision: "AgentDecision | None" = None
    next_step_index: int | None = None
    terminal_status: AgentTerminalStatus | None = None
    finalize_composite: bool = False
    is_dynamic: bool = False
    blocked_by: str | None = None
    replan_source: str = "none"
    deterministic_assessment: dict[str, Any] = field(default_factory=dict)
    llm_assessment: dict[str, Any] = field(default_factory=dict)
    rule_candidates: list[ReplanCandidate] = field(default_factory=list)
    llm_candidates: list[ReplanCandidate] = field(default_factory=list)
    selected_candidate: ReplanCandidate | None = None
    rejected_candidates: list[ReplanCandidate] = field(default_factory=list)
    selection_reason: str | None = None
    policy_boundary: str | None = None

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "should_continue": self.should_continue,
            "reason": self.reason,
            "decision": self.decision.to_event_payload() if self.decision else None,
            "next_step_index": self.next_step_index,
            "terminal_status": self.terminal_status.value if self.terminal_status else None,
            "finalize_composite": self.finalize_composite,
            "is_dynamic": self.is_dynamic,
            "blocked_by": self.blocked_by,
            "replan_source": self.replan_source,
            "deterministic_assessment": self.deterministic_assessment,
            "llm_assessment": self.llm_assessment,
            "rule_candidates": [item.to_event_payload() for item in self.rule_candidates],
            "llm_candidates": [item.to_event_payload() for item in self.llm_candidates],
            "selected_candidate": (
                self.selected_candidate.to_event_payload()
                if self.selected_candidate
                else None
            ),
            "rejected_candidates": [item.to_event_payload() for item in self.rejected_candidates],
            "selection_reason": self.selection_reason,
            "policy_boundary": self.policy_boundary,
        }


@dataclass
class AgentPolicyVerdict:
    allowed: bool
    category: str
    reason: str
    risk_level: str
    requires_approval: bool
    forbidden: bool
    policy_name: str = "agent_runtime_policy_v1"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "category": self.category,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
            "forbidden": self.forbidden,
            "policy_name": self.policy_name,
            "evidence": self.evidence,
        }


@dataclass
class AgentDecision:
    route_kind: str
    action_type: str
    action_name: str
    action_args: dict[str, Any]
    risk_level: str
    requires_approval: bool
    missing_fields: list[str]
    can_continue: bool
    reason: str
    confidence: float | None = None
    target: dict[str, Any] | None = None
    candidate_routes: list[str] = field(default_factory=list)
    decision_source: str = "intent_router"
    terminal_hint: str | None = None
    raw_decision: Any | None = None

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "route_kind": self.route_kind,
            "action_type": self.action_type,
            "action_name": self.action_name,
            "action_args": self.action_args,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
            "missing_fields": self.missing_fields,
            "can_continue": self.can_continue,
            "reason": self.reason,
            "confidence": self.confidence,
            "target": self.target,
            "candidate_routes": self.candidate_routes,
            "decision_source": self.decision_source,
            "terminal_hint": self.terminal_hint,
        }


@dataclass
class AgentAction:
    action_type: str
    action_name: str
    action_args: dict[str, Any]
    risk_level: str
    requires_approval: bool
    raw_decision: Any | None = None

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "action_name": self.action_name,
            "action_args": self.action_args,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
        }


@dataclass
class AgentObservation:
    status: str
    action: str | None
    route_kind: str | None
    output: dict[str, Any]
    natural_reply: str | None = None
    pending_id: int | None = None
    error_event_id: int | None = None

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "route_kind": self.route_kind,
            "pending_id": self.pending_id,
            "error_event_id": self.error_event_id,
            "output": self.output,
            "natural_reply": self.natural_reply,
        }


@dataclass
class AgentStep:
    index: int
    plan_step: AgentPlanStep | None = None
    decision: AgentDecision | None = None
    action: AgentAction | None = None
    policy_verdict: AgentPolicyVerdict | None = None
    observation: AgentObservation | None = None
    observation_assessment: ObservationAssessment | None = None
    replan_directive: ReplanDirective | ReplanResult | None = None
    terminal_status: AgentTerminalStatus | None = None

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "plan_step": self.plan_step.to_event_payload() if self.plan_step else None,
            "decision": self.decision.to_event_payload() if self.decision else None,
            "action": self.action.to_event_payload() if self.action else None,
            "policy_verdict": self.policy_verdict.to_event_payload() if self.policy_verdict else None,
            "observation": self.observation.to_event_payload() if self.observation else None,
            "observation_assessment": (
                self.observation_assessment.to_event_payload()
                if self.observation_assessment
                else None
            ),
            "replan_directive": (
                self.replan_directive.to_event_payload()
                if self.replan_directive
                else None
            ),
            "terminal_status": self.terminal_status.value if self.terminal_status else None,
        }


@dataclass
class AgentRunState:
    agent_run_id: str
    session_id: str
    user_message: str
    conversation_history: list[dict[str, Any]]
    conversation_id: str | None = None
    task_id: str | None = None
    task_state_version: int | None = None
    request_context: RuntimeRequestContext | None = None
    user_goal: str | None = None
    max_steps: int = 1
    step_index: int = 0
    context_snapshot: Any | None = None
    context_bundle: ContextBundle | None = None
    runtime_context: dict[str, Any] = field(default_factory=dict)
    context_history: list[dict[str, Any]] = field(default_factory=list)
    status_bar: dict[str, Any] = field(default_factory=dict)
    compression_records: list[dict[str, Any]] = field(default_factory=list)
    selected_skill_definitions: list[dict[str, Any]] = field(default_factory=list)
    skill_selection_trace: list[dict[str, Any]] = field(default_factory=list)
    model_input_reports: list[dict[str, Any]] = field(default_factory=list)
    context_budget_report: dict[str, Any] = field(default_factory=dict)
    steps: list[AgentStep] = field(default_factory=list)
    previous_observations: list[AgentObservation] = field(default_factory=list)
    current_db_snapshot: dict[str, Any] = field(default_factory=dict)
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    session_memory: dict[str, Any] = field(default_factory=dict)
    user_memories: list[dict[str, Any]] = field(default_factory=list)
    rag_evidence: list[dict[str, Any]] = field(default_factory=list)
    workflow_status: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    policy_history: list[dict[str, Any]] = field(default_factory=list)
    agent_plan: AgentPlan | None = None
    loop_plan: list[AgentDecision] = field(default_factory=list)
    loop_plan_index: int = 0
    loop_mode: str | None = None
    loop_stop_reason: str | None = None
    executed_action_signatures: list[str] = field(default_factory=list)
    initial_route_kind: str | None = None
    loop_plan_event_recorded: bool = False
    replan_history: list[dict[str, Any]] = field(default_factory=list)
    dynamic_plan_count: int = 0
    max_dynamic_replans: int = 2
    planning_context: dict[str, Any] = field(default_factory=dict)
    agentic_state_schema_version: int = 2
    agentic_mode: str | None = None
    safety_envelope: dict[str, Any] = field(default_factory=dict)
    resolved_tools: list[dict[str, Any]] = field(default_factory=list)
    v2_observations: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    candidate_plans: list[dict[str, Any]] = field(default_factory=list)
    model_turn_count: int = 0
    tool_call_count: int = 0
    compute_call_count: int = 0
    proposal_count: int = 0
    plan_patch_count: int = 0
    cumulative_model_tokens: int = 0
    agentic_started_at: float | None = None
    last_model_action: dict[str, Any] = field(default_factory=dict)
    terminal_status: AgentTerminalStatus | None = None
    final_response: ChatResponse | None = None
    last_error: dict[str, Any] | str | None = None


@dataclass(frozen=True)
class AgentRuntimeDeps:
    get_session_memory: Callable[..., list]
    save_session_memory: Callable[..., None]
    build_context_snapshot: Callable[..., Any]
    build_routing_decision: Callable[..., Any]
    dispatch_routing_decision: Callable[..., Any]
    normalize_input: Callable[..., Any]
    get_pending_form_state: Callable[..., dict[str, Any] | None]
    get_workflow_request_state: Callable[..., dict[str, Any] | None]
    tool_executor: Callable[..., Any]
    query_status_handler: Callable[..., Any]
    llm_handler: Callable[..., Any]
    append_decision_event: Callable[[dict[str, Any]], None]
@dataclass(frozen=True)
class RuntimeRequestContext:
    owner: str
    role: str
    workspace_id: str
    conversation_id: str
    task_id: str | None = None
    debug_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "role": self.role,
            "workspace_id": self.workspace_id,
            "conversation_id": self.conversation_id,
            "task_id": self.task_id,
            "debug_mode": self.debug_mode,
        }
