from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ContextProfile:
    name: str
    allowed_slices: tuple[str, ...]
    lane_weights: dict[str, float]
    allow_full_skills: bool = True


_PROFILES = {
    "router": ContextProfile(
        name="router",
        allowed_slices=("fact_context", "session_context", "user_memory_context", "tool_context", "policy_context"),
        lane_weights={"history": 0.10, "facts_policy": 0.25, "user_memory": 0.05, "observations": 0.05, "rag": 0.0, "skills": 0.20, "status": 0.05},
    ),
    "chat": ContextProfile(
        name="chat",
        allowed_slices=("fact_context", "session_context", "user_memory_context", "tool_context"),
        lane_weights={"history": 0.40, "facts_policy": 0.20, "user_memory": 0.10, "observations": 0.05, "rag": 0.0, "skills": 0.0, "status": 0.05},
        allow_full_skills=False,
    ),
    "lab_query": ContextProfile(
        name="lab_query",
        allowed_slices=("fact_context", "session_context", "user_memory_context", "tool_context", "policy_context"),
        lane_weights={"history": 0.15, "facts_policy": 0.40, "user_memory": 0.05, "observations": 0.10, "rag": 0.0, "skills": 0.05, "status": 0.05},
    ),
    "knowledge_query": ContextProfile(
        name="knowledge_query",
        allowed_slices=("session_context", "user_memory_context", "rag_context", "tool_context", "policy_context"),
        lane_weights={"history": 0.15, "facts_policy": 0.05, "user_memory": 0.10, "observations": 0.05, "rag": 0.40, "skills": 0.05, "status": 0.05},
    ),
    "workflow_write": ContextProfile(
        name="workflow_write",
        allowed_slices=("fact_context", "session_context", "user_memory_context", "tool_context", "policy_context"),
        lane_weights={"history": 0.15, "facts_policy": 0.35, "user_memory": 0.05, "observations": 0.10, "rag": 0.0, "skills": 0.15, "status": 0.05},
    ),
    "scientific_task": ContextProfile(
        name="scientific_task",
        allowed_slices=("fact_context", "session_context", "user_memory_context", "rag_context", "tool_context", "policy_context"),
        lane_weights={"history": 0.10, "facts_policy": 0.15, "user_memory": 0.05, "observations": 0.25, "rag": 0.25, "skills": 0.15, "status": 0.05},
    ),
    "replan": ContextProfile(
        name="replan",
        allowed_slices=("fact_context", "rag_context", "tool_context", "policy_context"),
        lane_weights={"history": 0.05, "facts_policy": 0.20, "user_memory": 0.0, "observations": 0.40, "rag": 0.10, "skills": 0.10, "status": 0.05},
    ),
}

CONTEXT_PROFILES = MappingProxyType(_PROFILES)


def resolve_context_profile(request_kind: str, route_kind: str | None = None) -> ContextProfile:
    if request_kind in CONTEXT_PROFILES:
        return CONTEXT_PROFILES[request_kind]
    route_aliases = {
        "query": "lab_query",
        "lab_query": "lab_query",
        "knowledge_query": "knowledge_query",
        "rag": "knowledge_query",
        "workflow": "workflow_write",
        "write_action": "workflow_write",
        "email": "workflow_write",
        "scientific_task": "scientific_task",
        "chat": "chat",
    }
    return CONTEXT_PROFILES[route_aliases.get(str(route_kind or ""), "chat")]
