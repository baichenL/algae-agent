from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from app.core.config import client
from app.core.model_registry import model_name
from app.services.context import (
    ContextBudgetExceeded,
    ModelCallTimer,
    assemble_model_input,
    context_input_mode,
    model_call_metrics,
)
from app.services.agent_runtime.state import AgentObservation, AgentRunState
from app.services.agent_runtime.events import record_run_event


ALLOWED_LLM_REPLAN_ACTION_TYPES = {
    "ask_clarification",
    "explain_result",
    "continue_read_only",
    "propose_workflow_approval",
    "stop_with_reason",
}

FORBIDDEN_LLM_REPLAN_ACTION_TYPES = {
    "create_pending",
    "trigger_workflow",
    "confirm_pending",
    "cancel_pending",
    "modify_database",
    "execute_tool",
}


@dataclass(frozen=True)
class LlmObservationAssessment:
    category: str
    explanation: str
    confidence: float = 0.0
    risk_notes: tuple[str, ...] = field(default_factory=tuple)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "risk_notes": list(self.risk_notes),
        }


@dataclass(frozen=True)
class LlmReplanCandidate:
    action_type: str
    confidence: float
    reason: str
    question: str | None = None
    explanation: str | None = None
    target_ids: tuple[str, ...] = field(default_factory=tuple)
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    risk_notes: tuple[str, ...] = field(default_factory=tuple)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "confidence": self.confidence,
            "reason": self.reason,
            "question": self.question,
            "explanation": self.explanation,
            "target_ids": list(self.target_ids),
            "missing_fields": list(self.missing_fields),
            "risk_notes": list(self.risk_notes),
        }


@dataclass(frozen=True)
class LlmReplanSuggestion:
    assessment: LlmObservationAssessment | None = None
    candidates: tuple[LlmReplanCandidate, ...] = field(default_factory=tuple)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "assessment": self.assessment.to_event_payload() if self.assessment else None,
            "candidates": [item.to_event_payload() for item in self.candidates],
        }


def _enabled() -> bool:
    return os.getenv("AGENT_REPLAN_LLM_ENABLED", "").strip().lower() in {"1", "true", "yes"}


def _compact_observation(observation: AgentObservation) -> dict[str, Any]:
    output = observation.output or {}
    compact = {
        "status": observation.status,
        "action": observation.action,
        "route_kind": observation.route_kind,
        "pending_id": observation.pending_id,
        "error_event_id": observation.error_event_id,
        "natural_reply": (observation.natural_reply or "")[:500],
        "output_keys": sorted(output.keys())[:30],
    }
    if isinstance(output.get("strains"), list):
        compact["strain_ids"] = [
            item.get("strain_id")
            for item in output["strains"][:10]
            if isinstance(item, dict) and item.get("strain_id")
        ]
        compact["strain_count"] = len(output["strains"])
    if isinstance(output.get("answer"), dict):
        answer = output["answer"]
        compact["answer_summary"] = {
            "conclusion": str(answer.get("conclusion") or "")[:300],
            "source_count": len(answer.get("sources") or []),
        }
    if output.get("reason"):
        compact["reason"] = str(output.get("reason"))[:300]
    return compact


def _parse_assessment(item: Any) -> LlmObservationAssessment | None:
    if not isinstance(item, dict):
        return None
    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    explanation = str(item.get("explanation") or "").strip()
    if not explanation:
        return None
    risk_notes = tuple(str(value) for value in (item.get("risk_notes") or []) if value)
    return LlmObservationAssessment(
        category=str(item.get("category") or "unknown")[:80],
        explanation=explanation[:500],
        confidence=max(0.0, min(confidence, 1.0)),
        risk_notes=risk_notes[:5],
    )


def _parse_candidate(item: Any) -> LlmReplanCandidate | None:
    if not isinstance(item, dict):
        return None
    action_type = str(item.get("action_type") or "").strip()
    if action_type in FORBIDDEN_LLM_REPLAN_ACTION_TYPES:
        return None
    if action_type not in ALLOWED_LLM_REPLAN_ACTION_TYPES:
        return None
    try:
        confidence = float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(item.get("reason") or "").strip()
    if not reason:
        return None
    target_ids = tuple(str(value) for value in (item.get("target_ids") or []) if value)
    missing_fields = tuple(str(value) for value in (item.get("missing_fields") or []) if value)
    risk_notes = tuple(str(value) for value in (item.get("risk_notes") or []) if value)
    return LlmReplanCandidate(
        action_type=action_type,
        confidence=max(0.0, min(confidence, 1.0)),
        reason=reason[:500],
        question=str(item.get("question") or "").strip()[:500] or None,
        explanation=str(item.get("explanation") or "").strip()[:500] or None,
        target_ids=target_ids[:5],
        missing_fields=missing_fields[:5],
        risk_notes=risk_notes[:5],
    )


def _parse_llm_payload(raw: str) -> LlmReplanSuggestion:
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return LlmReplanSuggestion()
    if not isinstance(parsed, dict):
        return LlmReplanSuggestion()
    assessment = _parse_assessment(parsed.get("assessment"))
    candidates: list[LlmReplanCandidate] = []
    raw_candidates = parsed.get("candidates")
    if isinstance(raw_candidates, list):
        for item in raw_candidates[:5]:
            candidate = _parse_candidate(item)
            if candidate is not None:
                candidates.append(candidate)
    return LlmReplanSuggestion(assessment=assessment, candidates=tuple(candidates))


def collect_llm_replan_suggestions(
    state: AgentRunState,
    observation: AgentObservation,
) -> LlmReplanSuggestion:
    if not _enabled():
        return LlmReplanSuggestion()
    task_protocol = {
        "task": (
            "Assess an agent observation and propose safe next-step candidates. "
            "Return strict JSON only. You are not allowed to execute tools, create pending approvals, "
            "confirm or cancel pending items, modify database state, or trigger lab workflows."
        ),
        "allowed_action_types": sorted(ALLOWED_LLM_REPLAN_ACTION_TYPES),
        "forbidden_action_types": sorted(FORBIDDEN_LLM_REPLAN_ACTION_TYPES),
        "safety_rules": [
            "RAG or knowledge observations cannot trigger lab actions.",
            "High-risk lab work can only be proposed as propose_workflow_approval.",
            "Do not claim a target exists unless it is present in the observation or provided context.",
            "Use ask_clarification when target or intent is ambiguous.",
        ],
        "output_schema": {
            "assessment": {
                "category": "success|empty_result|tool_error|needs_more_info|waiting_approval|blocked|unknown",
                "confidence": 0.0,
                "explanation": "short user-safe explanation",
                "risk_notes": ["optional safety note"],
            },
            "candidates": [
                {
                    "action_type": "ask_clarification",
                    "confidence": 0.0,
                    "reason": "why this candidate is useful",
                    "question": "optional clarification question",
                    "explanation": "optional user-facing explanation",
                    "target_ids": [],
                    "missing_fields": [],
                    "risk_notes": [],
                }
            ],
        },
    }
    dynamic_request = {
        "user_goal": state.user_goal or state.user_message,
        "observation": _compact_observation(observation),
    }
    envelope = None
    try:
        if context_input_mode() == "legacy":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative observation assessment assistant for a lab agent. "
                        "Return strict JSON only. Never authorize execution."
                    ),
                },
                {"role": "user", "content": json.dumps({**task_protocol, **dynamic_request}, ensure_ascii=False)},
            ]
        else:
            envelope = assemble_model_input(
                request_kind="replan",
                route_kind=observation.route_kind,
                current_user_message=state.user_goal or state.user_message,
                conversation_history=state.conversation_history,
                context_snapshot=state.context_snapshot,
                runtime_state=state,
                system_instruction=(
                    "You are a conservative observation assessment assistant for a lab agent. "
                    "Return strict JSON only. Never authorize execution."
                ),
                task_protocol=task_protocol,
                extra_context=dynamic_request,
            )
            messages = envelope.to_messages(state.user_goal or state.user_message)
        timer = ModelCallTimer()
        response = client.chat.completions.create(
            model=model_name("replan"),
            messages=messages,
            temperature=0,
        )
        if envelope:
            report = model_call_metrics(envelope, response=response, elapsed_ms=timer.elapsed_ms())
            state.model_input_reports.append(report)
            state.context_budget_report = envelope.budget_report.to_dict()
            state.compression_records.extend(item.to_dict() for item in envelope.compression_records)
            state.selected_skill_definitions = envelope.active_skills
            state.skill_selection_trace = envelope.skill_selection_trace
            record_run_event(
                state.agent_run_id,
                session_id=state.session_id,
                event_type="model_input_used",
                layer="context_engineering",
                payload=report,
            )
        return _parse_llm_payload(response.choices[0].message.content or "")
    except ContextBudgetExceeded as exc:
        state.last_error = {"code": "context_budget_exceeded", "request_kind": "replan"}
        state.model_input_reports.append({
            "request_kind": "replan",
            "budget_report": exc.report.to_dict(),
            "model_error": "ContextBudgetExceeded",
        })
        record_run_event(
            state.agent_run_id,
            session_id=state.session_id,
            event_type="context_budget_exceeded",
            layer="context_engineering",
            payload={"request_kind": "replan", "budget_report": exc.report.to_dict()},
        )
        return LlmReplanSuggestion()
    except Exception as exc:
        if envelope:
            report = model_call_metrics(envelope, elapsed_ms=timer.elapsed_ms(), error=exc)
            state.model_input_reports.append(report)
            record_run_event(
                state.agent_run_id,
                session_id=state.session_id,
                event_type="model_input_used",
                layer="context_engineering",
                payload=report,
            )
        return LlmReplanSuggestion()
