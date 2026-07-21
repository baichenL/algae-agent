from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from app.services.agent_runtime.state import (
    AgentRunState,
    AgentRuntimeDeps,
    ContextBundle,
    ContextSlice,
)
from app.services.agent_runtime.toolsets import toolset_for_context
from app.services.learning.selector import select_agent_learning_context
from app.services.context.status_bar import build_status_bar
from app.services.user_memory.extractor import extraction_enabled, shadow_mode
from app.services.user_memory.retriever import retrieve_user_memories
from app.services.user_memory.store import get_settings
from app.services.user_memory.catalog import allowed_predicates_for_route


@dataclass(frozen=True)
class AgentContextBuildResult:
    snapshot: Any
    event_payload: dict[str, Any]


def _safe_pending_action(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload") or {}
    data = payload.get("data") or {}
    return {
        "id": item.get("id"),
        "action_type": item.get("action_type"),
        "status": item.get("status"),
        "target": data.get("strain_id") or data.get("target") or "unknown",
        "requester": item.get("requester"),
    }


def _safe_strain(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "strain_id": item.get("strain_id"),
        "name_cn": item.get("name_cn"),
        "name_en": item.get("name_en"),
        "generation_number": item.get("generation_number"),
        "subculture_due": item.get("subculture_due"),
        "days_since_last_subculture": item.get("days_since_last_subculture"),
    }


def _snapshot_summary(snapshot: Any) -> dict[str, Any]:
    strains = list(getattr(snapshot, "strains", []) or [])
    pending_actions = list(getattr(snapshot, "pending_actions", []) or [])
    recent_experiments = list(getattr(snapshot, "recent_experiments", []) or [])
    target_strain = getattr(snapshot, "target_strain", None)
    target_pending_actions = list(getattr(snapshot, "target_pending_actions", []) or [])
    return {
        "session_id": getattr(snapshot, "session_id", None),
        "created_at": getattr(snapshot, "created_at", None),
        "strain_count": len(strains),
        "pending_count": len(pending_actions),
        "recent_experiment_count": len(recent_experiments),
        "target_strain_id": target_strain.get("strain_id") if isinstance(target_strain, dict) else None,
        "target_pending_count": len(target_pending_actions),
        "strains": [_safe_strain(item) for item in strains[:10] if isinstance(item, dict)],
        "pending_actions": [
            _safe_pending_action(item) for item in pending_actions[:10] if isinstance(item, dict)
        ],
        "selected_agent_memories": list(getattr(snapshot, "selected_agent_memories", []) or []),
        "selected_user_memories": list(getattr(snapshot, "selected_user_memories", []) or []),
        "selected_skill_summaries": list(getattr(snapshot, "selected_skill_summaries", []) or []),
        "learning_budget": dict(getattr(snapshot, "learning_budget", {}) or {}),
    }


def _session_memory_summary(state: AgentRunState) -> dict[str, Any]:
    return {
        "message_count": len(state.conversation_history),
        "last_user_message": state.user_message,
        "has_assistant_reply": any(
            item.get("role") == "assistant"
            for item in state.conversation_history
            if isinstance(item, dict)
        ),
    }


def _observation_summary(state: AgentRunState) -> list[dict[str, Any]]:
    return [
        {
            "status": item.status,
            "action": item.action,
            "route_kind": item.route_kind,
            "pending_id": item.pending_id,
            "error_event_id": item.error_event_id,
        }
        for item in state.previous_observations
    ]


def _rag_evidence(state: AgentRunState) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in state.previous_observations:
        if item.route_kind != "knowledge_query":
            continue
        output = item.output or {}
        answer = output.get("answer")
        if isinstance(answer, dict):
            evidence.append({
                "action": item.action,
                "conclusion": answer.get("conclusion"),
                "source_count": len(answer.get("sources") or []),
            })
        else:
            evidence.append({"action": item.action, "status": item.status})
    return evidence


def _workflow_status(snapshot: Any) -> dict[str, Any]:
    pending_actions = list(getattr(snapshot, "pending_actions", []) or [])
    workflow_pending = [
        item
        for item in pending_actions
        if isinstance(item, dict) and "workflow" in str(item.get("action_type", ""))
    ]
    return {
        "pending_workflow_count": len(workflow_pending),
        "pending_workflow_ids": [item.get("id") for item in workflow_pending[:10]],
    }


def _build_context_bundle(
    state: AgentRunState,
    db_snapshot: dict[str, Any],
) -> ContextBundle:
    fact_summary = {
        "strain_count": db_snapshot["strain_count"],
        "pending_count": db_snapshot["pending_count"],
        "recent_experiment_count": db_snapshot["recent_experiment_count"],
        "target_strain_id": db_snapshot["target_strain_id"],
    }
    session_summary = {
        "message_count": state.session_memory["message_count"],
        "has_assistant_reply": state.session_memory["has_assistant_reply"],
    }
    tool_summary = {
        "previous_observation_count": len(state.previous_observations),
        "last_error": state.last_error,
    }
    policy_summary = {
        "policy_count": len(state.policy_history),
        "missing_fields": state.missing_fields,
    }
    learning_payload = {
        key: value for key, value in (state.planning_context.get("learning") or {}).items()
        if key != "selected_user_memories"
    }
    return ContextBundle(
        fact_context=ContextSlice(
            name="fact_context",
            source="sqlite",
            authority="database_current_state",
            summary=fact_summary,
            payload={
                "strains": db_snapshot["strains"],
                "pending_actions": db_snapshot["pending_actions"],
                "workflow_status": state.workflow_status,
            },
        ),
        session_context=ContextSlice(
            name="session_context",
            source="session_memory",
            authority="conversation_history",
            summary=session_summary,
            payload={"last_user_message": state.session_memory["last_user_message"]},
        ),
        user_memory_context=ContextSlice(
            name="user_memory_context",
            source="user_memories",
            authority="user_preference_advisory",
            summary={"memory_count": len(state.user_memories)},
            payload={"memories": state.user_memories},
        ),
        rag_context=ContextSlice(
            name="rag_context",
            source="rag_observations",
            authority="knowledge_only",
            summary={"evidence_count": len(state.rag_evidence)},
            payload={"evidence": state.rag_evidence},
        ),
        tool_context=ContextSlice(
            name="tool_context",
            source="previous_observations",
            authority="current_run_observation",
            summary=tool_summary,
            payload={"observations": _observation_summary(state)},
        ),
        policy_context=ContextSlice(
            name="policy_context",
            source="agent_policy",
            authority="execution_boundary",
            summary=policy_summary,
            payload={
                "policy_history": state.policy_history,
                "learning_context": learning_payload,
            },
        ),
    )


def _build_snapshot(deps: AgentRuntimeDeps, session_id: str, strain_id: str | None) -> Any:
    if strain_id:
        try:
            return deps.build_context_snapshot(session_id, strain_id=strain_id)
        except TypeError:
            return deps.build_context_snapshot(session_id)
    return deps.build_context_snapshot(session_id)


def build_agent_context(
    state: AgentRunState,
    deps: AgentRuntimeDeps,
    *,
    strain_id: str | None = None,
    targeted: bool = False,
) -> AgentContextBuildResult:
    snapshot = _build_snapshot(deps, state.session_id, strain_id)
    learning = select_agent_learning_context(
        session_id=state.session_id,
        message=state.user_message,
        agent_run_id=state.agent_run_id,
    )
    try:
        snapshot = replace(
            snapshot,
            selected_agent_memories=learning.selected_agent_memories,
            selected_user_memories=learning.selected_user_memories,
            selected_skill_summaries=learning.selected_skill_summaries,
            learning_budget=learning.budget,
        )
    except Exception:
        try:
            setattr(snapshot, "selected_agent_memories", learning.selected_agent_memories)
            setattr(snapshot, "selected_user_memories", learning.selected_user_memories)
            setattr(snapshot, "selected_skill_summaries", learning.selected_skill_summaries)
            setattr(snapshot, "learning_budget", learning.budget)
        except Exception:
            pass
    db_snapshot = _snapshot_summary(snapshot)

    state.context_snapshot = snapshot
    state.user_goal = state.user_goal or state.user_message
    state.current_db_snapshot = db_snapshot
    state.pending_actions = db_snapshot["pending_actions"]
    state.session_memory = _session_memory_summary(state)
    state.user_memories = []
    request_context = state.request_context
    if request_context and extraction_enabled() and not shadow_mode():
        settings = get_settings(request_context.owner, request_context.workspace_id)
        if settings.get("enabled"):
            state.user_memories = retrieve_user_memories(
                owner_id=request_context.owner,
                workspace_id=request_context.workspace_id,
                query=state.user_message,
                route_kind=state.initial_route_kind or "router",
            )
    route = state.initial_route_kind or "router"
    legacy_enabled = os.getenv("MEMORY_LEGACY_USER_READ_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    if legacy_enabled and "general.note" in allowed_predicates_for_route(route):
        max_items = max(1, int(os.getenv("USER_MEMORY_MAX_ITEMS", "5")))
        char_budget = max(100, int(os.getenv("USER_MEMORY_CHAR_BUDGET", "1000")))
        used = sum(len(str(item.get("value") or "")) for item in state.user_memories)
        for legacy in learning.selected_user_memories:
            content = str(legacy.get("content") or "")[:500]
            if not content or len(state.user_memories) >= max_items or used + len(content) > char_budget:
                break
            state.user_memories.append({
                "id": f"legacy:{legacy.get('id')}",
                "memory_type": "note",
                "predicate": "general.note",
                "value": content,
                "revision": 0,
                "confidence": min(float(legacy.get("confidence") or 0.0), 0.5),
                "score": 0.05,
                "legacy": True,
            })
            used += len(content)
    state.rag_evidence = _rag_evidence(state)
    state.workflow_status = _workflow_status(snapshot)
    state.planning_context["learning"] = learning.to_dict()
    state.context_bundle = _build_context_bundle(state, db_snapshot)
    state.runtime_context = {
        "user_goal": state.user_goal,
        "step_index": state.step_index,
        "context_bundle": state.context_bundle.to_model_payload(),
        "previous_observations": _observation_summary(state),
        "current_db_snapshot": state.current_db_snapshot,
        "pending_actions": state.pending_actions,
        "session_memory": state.session_memory,
        "user_memories": state.user_memories,
        "rag_evidence": state.rag_evidence,
        "workflow_status": state.workflow_status,
        "missing_fields": state.missing_fields,
        "last_error": state.last_error,
        "policy_history": state.policy_history,
        "learning": state.planning_context["learning"],
        "request": state.request_context.to_dict() if state.request_context else None,
        "toolset": toolset_for_context(state.request_context),
    }
    state.status_bar = build_status_bar(state).to_dict()
    state.runtime_context["status_bar"] = state.status_bar

    event_payload = {
        "targeted": targeted,
        "strain_id": strain_id,
        "runtime_context": {
            "user_goal": state.runtime_context["user_goal"],
            "step_index": state.runtime_context["step_index"],
            "previous_observation_count": len(state.runtime_context["previous_observations"]),
            "missing_fields": state.runtime_context["missing_fields"],
            "last_error": state.runtime_context["last_error"],
            "policy_count": len(state.runtime_context["policy_history"]),
            "selected_agent_memory_count": len(learning.selected_agent_memories),
            "selected_user_memory_count": len(learning.selected_user_memories),
            "retrieved_user_memory_count": len(state.user_memories),
            "selected_skill_count": len(learning.selected_skill_summaries),
            "tool_count": len(state.runtime_context["toolset"]["tool_names"]),
            "status_bar": state.status_bar,
        },
        "context_bundle": state.context_bundle.to_event_payload(),
        "strain_count": db_snapshot["strain_count"],
        "pending_count": db_snapshot["pending_count"],
        "recent_experiment_count": db_snapshot["recent_experiment_count"],
        "target_pending_count": db_snapshot["target_pending_count"],
    }
    state.context_history.append(event_payload)
    return AgentContextBuildResult(snapshot=snapshot, event_payload=event_payload)
