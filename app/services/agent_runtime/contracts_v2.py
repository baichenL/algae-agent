from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

from app.core.effects import (
    CanonicalExecutionStatus,
    EffectClass,
    StateObservation,
    canonical_json,
    stable_hash,
)



SAFE_AGENT_EFFECTS = frozenset(
    {
        EffectClass.READ,
        EffectClass.COMPUTE,
        EffectClass.CONTROL_WRITE,
        EffectClass.ARTIFACT_WRITE,
        EffectClass.PROPOSAL_WRITE,
    }
)

TRUSTED_WORKER_EFFECTS = frozenset(
    {
        EffectClass.DOMAIN_FACT_COMMIT,
        EffectClass.EXTERNAL_ACTUATION,
    }
)


class ModelActionKind(str, Enum):
    TOOL_CALLS = "tool_calls"
    REQUEST_USER_INPUT = "request_user_input"
    FINAL_ANSWER = "final_answer"


class EvidenceAuthority(str, Enum):
    DOMAIN_FACT = "domain_fact"
    USER_INPUT = "user_input"
    TOOL_DERIVED = "tool_derived"
    RAG_ADVISORY = "rag_advisory"
    SIMULATION = "simulation"
    CONTROL_STATE = "control_state"


@dataclass(frozen=True)
class RuntimeBudgets:
    model_turns: int = 12
    tool_calls: int = 24
    compute_calls: int = 8
    proposals: int = 1
    candidates: int = 3
    plan_patches: int = 3
    wall_time_seconds: int = 180
    cumulative_model_tokens: int = 128_000
    observation_inline_tokens: int = 4_000
    tool_schema_tokens: int = 8_000

    @classmethod
    def from_env(cls) -> "RuntimeBudgets":
        def _integer(name: str, default: int) -> int:
            return max(1, int(os.getenv(name, str(default))))

        return cls(
            model_turns=_integer("AGENT_MAX_MODEL_TURNS", 12),
            tool_calls=_integer("AGENT_MAX_TOOL_CALLS", 24),
            compute_calls=_integer("AGENT_MAX_COMPUTE_CALLS", 8),
            proposals=_integer("AGENT_MAX_PROPOSALS", 1),
            candidates=_integer("AGENT_MAX_CANDIDATES", 3),
            plan_patches=_integer("AGENT_MAX_PLAN_PATCHES", 3),
            wall_time_seconds=_integer("AGENT_WALL_TIME_SECONDS", 180),
            cumulative_model_tokens=_integer("AGENT_CUMULATIVE_MODEL_TOKENS", 128_000),
            observation_inline_tokens=_integer("AGENT_OBSERVATION_INLINE_TOKENS", 4_000),
            tool_schema_tokens=_integer("AGENT_TOOL_SCHEMA_TOKENS", 8_000),
        )


@dataclass(frozen=True)
class SafetyEnvelope:
    principal_id: str
    workspace_id: str
    role: str
    control_flow: str
    allowed_effect_classes: tuple[str, ...]
    denied_capabilities: tuple[str, ...]
    policy_version: str
    budgets: RuntimeBudgets
    created_at_epoch: float
    expires_at_epoch: float
    envelope_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def allows(self, effect: EffectClass | str) -> bool:
        value = effect.value if isinstance(effect, EffectClass) else str(effect)
        return value in self.allowed_effect_classes and value not in self.denied_capabilities

    def is_expired(self, *, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at_epoch

    def to_event_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["budgets"] = asdict(self.budgets)
        return payload


@dataclass(frozen=True)
class ResolvedTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    effect_class: str
    risk_level: str
    requires_approval: bool
    executor_kind: str
    parallel_safe: bool
    timeout_seconds: int
    result_authority: str

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class ModelToolCall:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelAction:
    kind: ModelActionKind
    tool_calls: tuple[ModelToolCall, ...] = ()
    content: str | None = None
    question: str | None = None
    reason_summary: str | None = None


@dataclass(frozen=True)
class EvidenceRef:
    ref_id: str
    authority: str
    source: str
    resource_version: str | None = None
    summary: str | None = None


@dataclass(frozen=True)
class ToolObservation:
    observation_id: str
    tool_call_id: str
    tool_name: str
    status: str
    effect_class: str
    authority: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[EvidenceRef, ...] = ()
    error_code: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    retryable: bool = False
    suggested_repairs: tuple[str, ...] = ()
    pending_id: int | None = None
    resource_versions: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    artifact_ref: str | None = None
    duration_ms: int = 0

    def to_model_payload(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence_refs": [asdict(item) for item in self.evidence_refs],
        }


@dataclass
class Hypothesis:
    hypothesis_id: str
    claim: str
    status: Literal["active", "supported", "weakened", "rejected"]
    supporting_evidence_refs: list[str] = field(default_factory=list)
    contradicting_evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5
    uncertainties: list[str] = field(default_factory=list)
    next_best_test: str | None = None
    created_step: int = 0
    updated_step: int = 0


@dataclass
class CandidatePlan:
    candidate_id: str
    version: int
    design: dict[str, Any]
    evidence_refs: list[str] = field(default_factory=list)
    hypothesis_refs: list[str] = field(default_factory=list)
    validation_result: dict[str, Any] = field(default_factory=dict)
    simulation_result: dict[str, Any] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    uncertainties: list[str] = field(default_factory=list)
    status: str = "draft"


@dataclass(frozen=True)
class ProposalReadiness:
    ready: bool
    missing_requirements: tuple[str, ...] = ()
    conflicting_evidence: tuple[str, ...] = ()
    suggested_next_investigations: tuple[str, ...] = ()


@dataclass
class AgenticRunState:
    safety_envelope: SafetyEnvelope
    model_turn_count: int = 0
    tool_call_count: int = 0
    compute_call_count: int = 0
    proposal_count: int = 0
    plan_patch_count: int = 0
    cumulative_model_tokens: int = 0
    observations: list[ToolObservation] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    candidates: list[CandidatePlan] = field(default_factory=list)
    executed_signatures: list[str] = field(default_factory=list)
    started_at_epoch: float = field(default_factory=time.time)

    def remaining_budget(self) -> dict[str, int]:
        budgets = self.safety_envelope.budgets
        return {
            "model_turns": max(0, budgets.model_turns - self.model_turn_count),
            "tool_calls": max(0, budgets.tool_calls - self.tool_call_count),
            "compute_calls": max(0, budgets.compute_calls - self.compute_call_count),
            "proposals": max(0, budgets.proposals - self.proposal_count),
            "plan_patches": max(0, budgets.plan_patches - self.plan_patch_count),
            "model_tokens": max(0, budgets.cumulative_model_tokens - self.cumulative_model_tokens),
        }

    def budget_exhausted(self) -> str | None:
        remaining = self.remaining_budget()
        for name, value in remaining.items():
            if value <= 0:
                return name
        if time.time() - self.started_at_epoch >= self.safety_envelope.budgets.wall_time_seconds:
            return "wall_time"
        return None
