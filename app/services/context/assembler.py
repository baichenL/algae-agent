from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from app.services.context.budget import (
    ContextBudgetExceeded,
    ContextBudgetReport,
    canonical_json,
    estimate_tokens,
    hard_budget_ratio,
    input_token_budget,
    soft_budget_ratio,
    stable_hash,
)
from app.services.context.compression import (
    CompressionRecord,
    compress_history,
    compress_observations,
    protected_view,
    split_current_turn,
    trim_lane,
)
from app.services.context.profiles import ContextProfile, resolve_context_profile
from app.services.context.status_bar import AgentStatusBar, build_status_bar
from app.services.skills import (
    select_skill_definitions,
    skill_manifest_for_router,
    skill_manifest_version,
)


PROMPT_VERSION = "algae-agent-context/v2"


def context_input_mode() -> str:
    return "legacy" if os.getenv("CONTEXT_INPUT_MODE", "assembled").strip().lower() == "legacy" else "assembled"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: cleaned for key, item in sorted(value.items()) if (cleaned := _clean(item)) not in (None, "", [], {})}
    if isinstance(value, (list, tuple)):
        return [cleaned for item in value if (cleaned := _clean(item)) not in (None, "", [], {})]
    return value


def _snapshot_context(snapshot: Any | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    target = getattr(snapshot, "target_strain", None)
    return _clean({
        "fact_context": {
            "source": "sqlite",
            "authority": "database_current_state",
            "payload": {
                "target_strain": target,
                "strains": list(getattr(snapshot, "strains", []) or [])[:20],
                "pending_actions": list(getattr(snapshot, "pending_actions", []) or [])[:20],
                "recent_experiments": list(getattr(snapshot, "recent_experiments", []) or [])[:10],
                "task_frame": getattr(snapshot, "task_frame", None),
            },
        },
        "session_context": {
            "source": "session_memory",
            "authority": "conversation_history",
            "payload": {
                "selected_agent_memories": list(getattr(snapshot, "selected_agent_memories", []) or []),
            },
        },
        "user_memory_context": {
            "source": "legacy_session_memory",
            "authority": "user_preference_advisory",
            "payload": {
                "memories": list(getattr(snapshot, "selected_user_memories", []) or []),
            },
        },
    })


def _runtime_context(state: Any | None, snapshot: Any | None) -> dict[str, Any]:
    bundle = getattr(state, "context_bundle", None) if state is not None else None
    if bundle is not None and hasattr(bundle, "to_model_payload"):
        return _clean(bundle.to_model_payload())
    if bundle is not None and hasattr(bundle, "to_event_payload"):
        return _clean(bundle.to_event_payload())
    runtime = dict(getattr(state, "runtime_context", {}) or {}) if state is not None else {}
    if isinstance(runtime.get("context_bundle"), dict):
        return _clean(runtime["context_bundle"])
    return _snapshot_context(snapshot)


def _route_context(context: dict[str, Any], profile: ContextProfile) -> dict[str, Any]:
    return {name: context[name] for name in profile.allowed_slices if name in context}


def _tool_version(tool_manifest: list[dict[str, Any]]) -> str:
    return stable_hash(tool_manifest)[:16]


@dataclass(frozen=True)
class ContextAssemblyFeatures:
    status_bar: bool = True
    dynamic_skills: bool = True
    compression: bool = True


@dataclass(frozen=True)
class ModelInputEnvelope:
    prompt_version: str
    request_kind: str
    route_kind: str | None
    stable_messages: list[dict[str, str]]
    history_messages: list[dict[str, str]]
    dynamic_context: dict[str, Any]
    active_skills: list[dict[str, Any]]
    status_bar: AgentStatusBar
    budget_report: ContextBudgetReport
    compression_records: list[CompressionRecord] = field(default_factory=list)
    skill_selection_trace: list[dict[str, Any]] = field(default_factory=list)
    status_bar_enabled: bool = True

    def to_messages(self, current_user_message: str) -> list[dict[str, str]]:
        messages = [*self.stable_messages, *self.history_messages]
        if self.dynamic_context:
            messages.append({
                "role": "system",
                "content": "Authoritative dynamic context: " + canonical_json(self.dynamic_context),
            })
        if self.active_skills:
            messages.append({
                "role": "system",
                "content": "Selected skill procedures (advisory; policy remains authoritative): " + canonical_json(self.active_skills),
            })
        if self.status_bar_enabled:
            messages.append({"role": "system", "content": "Agent status bar: " + self.status_bar.render()})
        messages.append({"role": "user", "content": current_user_message})
        return messages

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            "request_kind": self.request_kind,
            "route_kind": self.route_kind,
            "budget_report": self.budget_report.to_dict(),
            "status_bar": self.status_bar.to_dict(),
            "active_skills": [f"{item.get('name')}@{item.get('version')}" for item in self.active_skills],
            "skill_selection_trace": self.skill_selection_trace,
            "compression_records": [item.to_dict() for item in self.compression_records],
        }


def _lane_values(dynamic_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "facts_policy": {
            key: value for key, value in dynamic_context.items() if key in {"fact_context", "session_context", "policy_context", "request"}
        },
        "observations": dynamic_context.get("tool_context") or {},
        "rag": dynamic_context.get("rag_context") or {},
        "user_memory": dynamic_context.get("user_memory_context") or {},
    }


def _rebuild_dynamic(lanes: dict[str, Any]) -> dict[str, Any]:
    dynamic: dict[str, Any] = {}
    for lane in ("facts_policy", "observations", "rag", "user_memory"):
        value = lanes.get(lane)
        if not value:
            continue
        if lane == "facts_policy" and isinstance(value, dict):
            dynamic.update(value)
        elif lane == "observations":
            dynamic["tool_context"] = value
        elif lane == "rag":
            dynamic["rag_context"] = value
        elif lane == "user_memory":
            dynamic["user_memory_context"] = value
    return _clean(dynamic)


def assemble_model_input(
    *,
    request_kind: str,
    current_user_message: str,
    system_instruction: str,
    task_protocol: dict[str, Any],
    route_kind: str | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    context_snapshot: Any | None = None,
    runtime_state: Any | None = None,
    extra_context: dict[str, Any] | None = None,
    tool_manifest: list[dict[str, Any]] | None = None,
    requested_effect: str | None = None,
    entity_mentions: list[str] | tuple[str, ...] | None = None,
    features: ContextAssemblyFeatures | None = None,
) -> ModelInputEnvelope:
    features = features or ContextAssemblyFeatures()
    profile = resolve_context_profile(request_kind, route_kind)
    tools = sorted(tool_manifest or [], key=lambda item: canonical_json(item))
    manifest = [
        item for item in skill_manifest_for_router()
        if item.get("route_kind") in {route_kind, profile.name} or request_kind == "router"
    ]
    stable_protocol = _clean({
        "prompt_version": PROMPT_VERSION,
        "request_kind": request_kind,
        "route_kind": route_kind,
        "profile": profile.name,
        "task_protocol": task_protocol,
        "toolset_version": _tool_version(tools),
        "tools": tools,
        "skill_manifest_version": skill_manifest_version(),
        "skill_manifest": manifest,
    })
    stable_messages = [
        {"role": "system", "content": f"[{PROMPT_VERSION}] {system_instruction}"},
        {"role": "system", "content": canonical_json(stable_protocol)},
    ]
    prefix_hash = stable_hash(stable_messages)

    history = split_current_turn(conversation_history or [], current_user_message)
    records: list[CompressionRecord] = []
    if features.compression:
        history, records = compress_history(history)
    routed = _route_context(_runtime_context(runtime_state, context_snapshot), profile)
    if extra_context:
        routed["request"] = _clean(extra_context)

    max_skills = max(0, int(os.getenv("CONTEXT_MAX_ACTIVE_SKILLS", "2")))
    active_skills: list[dict[str, Any]] = []
    skill_trace: list[dict[str, Any]] = []
    if features.dynamic_skills and profile.allow_full_skills and max_skills:
        active_skills, skill_trace = select_skill_definitions(
            route_kind=route_kind or profile.name,
            message=current_user_message,
            requested_effect=requested_effect,
            entity_mentions=entity_mentions,
            max_skills=max_skills,
        )

    tool_context = routed.get("tool_context")
    if isinstance(tool_context, dict):
        payload = tool_context.get("payload") if isinstance(tool_context.get("payload"), dict) else tool_context
        observations = payload.get("observations") if isinstance(payload, dict) else None
        if features.compression and isinstance(observations, list):
            compressed, observation_records = compress_observations(observations)
            payload["observations"] = compressed
            records.extend(observation_records)

    selected_names = [f"{item.get('name')}@{item.get('version')}" for item in active_skills]
    status_bar = build_status_bar(
        runtime_state,
        goal=current_user_message,
        route_kind=route_kind,
        snapshot=context_snapshot,
        selected_skills=selected_names,
    )

    budget = input_token_budget()
    soft_limit = int(budget * soft_budget_ratio())
    hard_limit = int(budget * hard_budget_ratio())
    lanes = _lane_values(routed)
    lane_tokens = {
        "history": estimate_tokens(history),
        "facts_policy": estimate_tokens(lanes["facts_policy"]),
        "observations": estimate_tokens(lanes["observations"]),
        "rag": estimate_tokens(lanes["rag"]),
        "user_memory": estimate_tokens(lanes["user_memory"]),
        "skills": estimate_tokens(active_skills),
        "status": estimate_tokens(status_bar.to_dict()) if features.status_bar else 0,
    }
    initial_total = estimate_tokens(stable_messages) + sum(lane_tokens.values()) + estimate_tokens(current_user_message)
    if features.compression and initial_total > soft_limit:
        for lane in ("user_memory", "rag", "observations", "facts_policy"):
            weight = profile.lane_weights.get(lane, 0.0)
            lane_limit = max(64, int(budget * weight)) if weight else 64
            lanes[lane], lane_records = trim_lane(lanes[lane], lane_limit, lane=lane)
            records.extend(lane_records)
        routed = _rebuild_dynamic(lanes)
        history_limit = max(128, int(budget * profile.lane_weights.get("history", 0.1)))
        if estimate_tokens(history) > history_limit:
            history = history[-2:]

    protected = {
        "status_bar": status_bar.to_dict() if features.status_bar else {},
        "dynamic": protected_view(routed),
        "current_user_message": current_user_message,
    }
    protected_tokens = estimate_tokens(protected)
    final_lane_tokens = {
        "history": estimate_tokens(history),
        "facts_policy": estimate_tokens(_lane_values(routed)["facts_policy"]),
        "observations": estimate_tokens(_lane_values(routed)["observations"]),
        "rag": estimate_tokens(_lane_values(routed)["rag"]),
        "user_memory": estimate_tokens(_lane_values(routed)["user_memory"]),
        "skills": estimate_tokens(active_skills),
        "status": estimate_tokens(status_bar.to_dict()) if features.status_bar else 0,
    }
    total = estimate_tokens(stable_messages) + sum(final_lane_tokens.values()) + estimate_tokens(current_user_message)
    report = ContextBudgetReport(
        token_budget=budget,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
        estimated_input_tokens=total,
        lane_tokens=final_lane_tokens,
        stable_prefix_tokens=estimate_tokens(stable_messages),
        prefix_hash=prefix_hash,
        compression_applied=bool(records),
        protected_tokens=protected_tokens,
    )
    if protected_tokens > hard_limit or total > budget:
        raise ContextBudgetExceeded(report)
    return ModelInputEnvelope(
        prompt_version=PROMPT_VERSION,
        request_kind=request_kind,
        route_kind=route_kind,
        stable_messages=stable_messages,
        history_messages=history,
        dynamic_context=routed,
        active_skills=active_skills,
        status_bar=status_bar,
        budget_report=report,
        compression_records=records,
        skill_selection_trace=skill_trace,
        status_bar_enabled=features.status_bar,
    )


def model_call_metrics(
    envelope: ModelInputEnvelope,
    *,
    response: Any | None = None,
    elapsed_ms: float | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    prompt_tokens = (
        usage.get("prompt_tokens") if isinstance(usage, dict) else getattr(usage, "prompt_tokens", None)
    ) if usage is not None else None
    details = (
        usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
    ) if usage is not None else None
    cached_tokens = (
        details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
    ) if details is not None else None
    return {
        **envelope.to_trace_payload(),
        "actual_prompt_tokens": prompt_tokens,
        "cached_input_tokens": cached_tokens,
        "elapsed_ms": round(elapsed_ms, 3) if elapsed_ms is not None else None,
        "model_error": error.__class__.__name__ if error else None,
    }


class ModelCallTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000
