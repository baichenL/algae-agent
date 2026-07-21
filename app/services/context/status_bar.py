from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.context.budget import canonical_json


@dataclass(frozen=True)
class AgentStatusBar:
    goal: str
    route_kind: str | None = None
    step_index: int = 0
    max_steps: int = 1
    target_ids: list[str] = field(default_factory=list)
    workflow_status: str | None = None
    pending_ids: list[int] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    last_action: str | None = None
    last_observation_status: str | None = None
    last_error: str | None = None
    selected_skills: list[str] = field(default_factory=list)
    evidence_state: str = "not_applicable"
    remaining_replans: int = 0
    stop_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        values = {
            "goal": self.goal,
            "route_kind": self.route_kind,
            "step": f"{self.step_index}/{self.max_steps}",
            "target_ids": self.target_ids,
            "workflow_status": self.workflow_status,
            "pending_ids": self.pending_ids,
            "missing_fields": self.missing_fields,
            "last_action": self.last_action,
            "last_observation_status": self.last_observation_status,
            "last_error": self.last_error,
            "selected_skills": self.selected_skills,
            "evidence_state": self.evidence_state,
            "remaining_replans": self.remaining_replans,
            "stop_conditions": self.stop_conditions,
        }
        return {key: value for key, value in values.items() if value not in (None, [], "")}

    def render(self) -> str:
        return canonical_json(self.to_dict())[:1000]


def build_status_bar(
    state: Any | None = None,
    *,
    goal: str = "",
    route_kind: str | None = None,
    snapshot: Any | None = None,
    selected_skills: list[str] | None = None,
) -> AgentStatusBar:
    if state is None:
        pending = list(getattr(snapshot, "pending_actions", []) or [])
        pending_ids = [int(item["id"]) for item in pending if isinstance(item, dict) and item.get("id") is not None]
        target = getattr(snapshot, "target_strain", None)
        target_ids = [str(target.get("strain_id"))] if isinstance(target, dict) and target.get("strain_id") else []
        return AgentStatusBar(
            goal=goal,
            route_kind=route_kind,
            target_ids=target_ids,
            pending_ids=pending_ids[:10],
            selected_skills=selected_skills or [],
            workflow_status="waiting_approval" if pending_ids else None,
            stop_conditions=["approval_required"] if pending_ids else [],
        )

    observations = list(getattr(state, "previous_observations", []) or [])
    last = observations[-1] if observations else None
    pending = list(getattr(state, "pending_actions", []) or [])
    pending_ids = [int(item["id"]) for item in pending if isinstance(item, dict) and item.get("id") is not None]
    target_ids: list[str] = []
    snapshot_target = getattr(getattr(state, "context_snapshot", None), "target_strain", None)
    if isinstance(snapshot_target, dict) and snapshot_target.get("strain_id"):
        target_ids.append(str(snapshot_target["strain_id"]))
    workflow = dict(getattr(state, "workflow_status", {}) or {})
    terminal = getattr(state, "terminal_status", None)
    terminal_value = getattr(terminal, "value", terminal)
    stop_conditions = []
    if pending_ids or terminal_value == "waiting_approval":
        stop_conditions.append("approval_required")
    if int(getattr(state, "step_index", 0)) >= int(getattr(state, "max_steps", 1)):
        stop_conditions.append("max_steps")
    if terminal_value:
        stop_conditions.append(f"terminal:{terminal_value}")
    return AgentStatusBar(
        goal=str(getattr(state, "user_goal", None) or getattr(state, "user_message", None) or goal),
        route_kind=route_kind or getattr(state, "initial_route_kind", None),
        step_index=int(getattr(state, "step_index", 0)),
        max_steps=max(1, int(getattr(state, "max_steps", 1))),
        target_ids=target_ids,
        workflow_status="waiting_approval" if pending_ids else ("active" if workflow.get("pending_workflow_count") else None),
        pending_ids=pending_ids[:10],
        missing_fields=list(getattr(state, "missing_fields", []) or []),
        last_action=getattr(last, "action", None),
        last_observation_status=getattr(last, "status", None),
        last_error=getattr(state, "last_error", None),
        selected_skills=selected_skills or list(getattr(state, "selected_skill_names", []) or []),
        evidence_state="available" if getattr(state, "rag_evidence", None) else "not_applicable",
        remaining_replans=max(0, int(getattr(state, "max_dynamic_replans", 0)) - int(getattr(state, "dynamic_plan_count", 0))),
        stop_conditions=stop_conditions,
    )
