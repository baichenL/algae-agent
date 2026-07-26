from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class RouteKind(str, Enum):
    POLICY_REJECTED = "policy_rejected"
    TOOL_INFO = "tool_info"
    EMAIL = "email"
    LAB_QUERY = "lab_query"
    PENDING_QUERY = "pending_query"
    PENDING_FORM = "pending_form"
    KNOWLEDGE_QUERY = "knowledge_query"
    WRITE_ACTION = "write_action"
    WORKFLOW = "workflow"
    WORKFLOW_AUDIT = "workflow_audit"
    CHAT = "chat"
    CLARIFICATION = "clarification"
    COMPOSITE = "composite"
    SCIENTIFIC_TASK = "scientific_task"


class EffectKind(str, Enum):
    """Observable effect produced by a route, independent of its business label."""

    READ = "read"
    DRAFT = "draft"
    PROPOSE = "propose"
    EXECUTE = "execute"


class SpeechAct(str, Enum):
    COMMAND = "command"
    DIAGNOSTIC = "diagnostic"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    SUSPEND = "suspend"
    SLOT_VALUE = "slot_value"
    QUERY = "query"


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReasonCode(str, Enum):
    POLICY_CAPABILITY_FORBIDDEN = "policy_capability_forbidden"
    TOOL_INFO_MATCHED = "tool_info_matched"
    EMAIL_MATCHED = "email_matched"
    LAB_QUERY_MATCHED = "lab_query_matched"
    PENDING_QUERY_MATCHED = "pending_query_matched"
    PENDING_FORM_CANCEL = "pending_form_cancel"
    PENDING_FORM_CONTINUATION = "pending_form_continuation"
    PENDING_FORM_CONFLICT = "pending_form_conflict"
    KNOWLEDGE_QUERY_MATCHED = "knowledge_query_matched"
    WRITE_ACTION_MATCHED = "write_action_matched"
    WORKFLOW_EXPLICIT_EXECUTION = "workflow_explicit_execution"
    WORKFLOW_TARGET_REQUIRED = "workflow_target_required"
    WORKFLOW_SLOT_CONTINUATION = "workflow_slot_continuation"
    WORKFLOW_REQUEST_CANCEL = "workflow_request_cancel"
    WORKFLOW_AUDIT_QUERY = "workflow_audit_query"
    WORKFLOW_CONFIRMATION_REQUIRES_UI = "workflow_confirmation_requires_ui"
    AMBIGUOUS_SUBCULTURE = "ambiguous_subculture"
    DEFAULT_CHAT = "default_chat"
    MULTIPLE_INTENTS_CONFLICT = "multiple_intents_conflict"
    MULTIPLE_TARGETS_CONFLICT = "multiple_targets_conflict"
    TARGET_NOT_FOUND = "target_not_found"
    REQUIRED_FIELDS_MISSING = "required_fields_missing"
    COMPOSITE_PLAN = "composite_plan"
    SCIENTIFIC_TASK_MATCHED = "scientific_task_matched"


class IntentFrame(BaseModel):
    """A semantic proposal from the LLM, never an authorization decision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["intent-frame/v1"] = "intent-frame/v1"
    goal: str = Field(min_length=1, max_length=1000)
    primary_route: RouteKind
    alternatives: tuple[RouteKind, ...] = ()
    speech_act: SpeechAct
    requested_effect: EffectKind
    entity_mentions: tuple[str, ...] = ()
    slots: dict[str, Any] = Field(default_factory=dict)
    missing_slots: tuple[str, ...] = ()
    suggested_skill: str | None = None
    suggested_tools: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedInput:
    original_text: str
    normalized_text: str
    match_text: str


@dataclass(frozen=True)
class RoutingInput:
    message: NormalizedInput
    session_id: str
    context_snapshot: Any
    active_pending_form: dict[str, Any] | None = None
    active_workflow_request: dict[str, Any] | None = None


@dataclass(frozen=True)
class EntityRef:
    entity_type: Literal["strain"]
    canonical_id: str | None
    mention: str
    source: Literal["explicit_id", "database_id", "database_name", "latin_name"]
    exists: bool


@dataclass(frozen=True)
class RouteCandidate:
    kind: RouteKind
    reason_code: ReasonCode
    risk_level: RiskLevel
    evidence: str
    source: Literal["rule", "llm", "entity", "system"] = "rule"
    score: float = 1.0
    evidence_items: tuple[str, ...] = field(default_factory=tuple)
    negative_signals: tuple[str, ...] = field(default_factory=tuple)
    selection_reason: str | None = None
    speech_act: SpeechAct | None = None
    target: EntityRef | None = None
    entity_options: tuple[EntityRef, ...] = field(default_factory=tuple)
    query_type: str | None = None
    operation: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    form_candidates: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.evidence_items:
            object.__setattr__(self, "evidence_items", (self.evidence,))


@dataclass(frozen=True, kw_only=True)
class BaseDecision:
    kind: RouteKind
    reason_code: ReasonCode
    risk_level: RiskLevel
    explanation: str
    source_text: str = ""
    target: EntityRef | None = None
    candidates: tuple[RouteCandidate, ...] = field(default_factory=tuple)
    selection_trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ToolInfoDecision(BaseDecision):
    kind: Literal[RouteKind.TOOL_INFO] = RouteKind.TOOL_INFO


@dataclass(frozen=True, kw_only=True)
class EmailDecision(BaseDecision):
    request_spec: dict[str, Any] = field(default_factory=dict)
    kind: Literal[RouteKind.EMAIL] = RouteKind.EMAIL


@dataclass(frozen=True, kw_only=True)
class QueryDecision(BaseDecision):
    query_type: str
    targets: tuple[EntityRef, ...] = field(default_factory=tuple)
    kind: RouteKind = RouteKind.LAB_QUERY


@dataclass(frozen=True, kw_only=True)
class PendingFormDecision(BaseDecision):
    form_action: Literal["cancel", "suspend", "continue", "conflict", "start"]
    operation: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    form_candidates: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    kind: Literal[RouteKind.PENDING_FORM] = RouteKind.PENDING_FORM


@dataclass(frozen=True, kw_only=True)
class KnowledgeDecision(BaseDecision):
    kind: Literal[RouteKind.KNOWLEDGE_QUERY] = RouteKind.KNOWLEDGE_QUERY


@dataclass(frozen=True, kw_only=True)
class WriteDecision(BaseDecision):
    operation: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    kind: Literal[RouteKind.WRITE_ACTION] = RouteKind.WRITE_ACTION


@dataclass(frozen=True, kw_only=True)
class WorkflowDecision(BaseDecision):
    speech_act: SpeechAct = SpeechAct.COMMAND
    workflow_name: str = "subculture"
    kind: Literal[RouteKind.WORKFLOW] = RouteKind.WORKFLOW


@dataclass(frozen=True, kw_only=True)
class WorkflowAuditDecision(BaseDecision):
    speech_act: SpeechAct = SpeechAct.DIAGNOSTIC
    pending_id: int | None = None
    workflow_name: str = "subculture"
    kind: Literal[RouteKind.WORKFLOW_AUDIT] = RouteKind.WORKFLOW_AUDIT


@dataclass(frozen=True, kw_only=True)
class ScientificTaskDecision(BaseDecision):
    arguments: dict[str, Any] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    kind: Literal[RouteKind.SCIENTIFIC_TASK] = RouteKind.SCIENTIFIC_TASK


@dataclass(frozen=True, kw_only=True)
class ChatDecision(BaseDecision):
    kind: Literal[RouteKind.CHAT] = RouteKind.CHAT


@dataclass(frozen=True, kw_only=True)
class ForbiddenDecision(BaseDecision):
    capability_request: dict[str, Any] = field(default_factory=dict)
    kind: Literal[RouteKind.POLICY_REJECTED] = RouteKind.POLICY_REJECTED


@dataclass(frozen=True, kw_only=True)
class ClarificationDecision(BaseDecision):
    questions: tuple[str, ...] = field(default_factory=tuple)
    response_action: str = "routing_clarification"
    kind: Literal[RouteKind.CLARIFICATION] = RouteKind.CLARIFICATION


@dataclass(frozen=True, kw_only=True)
class CompositeDecision(BaseDecision):
    """Ordered, independently dispatchable intentions from one user turn."""

    steps: tuple[BaseDecision, ...] = field(default_factory=tuple)
    kind: Literal[RouteKind.COMPOSITE] = RouteKind.COMPOSITE


RoutingDecision: TypeAlias = (
    ForbiddenDecision
    | ToolInfoDecision
    | EmailDecision
    | QueryDecision
    | PendingFormDecision
    | KnowledgeDecision
    | WriteDecision
    | WorkflowDecision
    | WorkflowAuditDecision
    | ScientificTaskDecision
    | ChatDecision
    | ClarificationDecision
    | CompositeDecision
)
