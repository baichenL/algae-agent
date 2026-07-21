from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app.services.learning.store import (
    MEMORY_CHAR_BUDGET,
    SKILL_SUMMARY_CHAR_BUDGET,
    list_agent_memories,
    list_agent_skills,
    record_learning_usage,
)
from app.services.user_memory.catalog import contains_prohibited_sensitive_data


@dataclass(frozen=True)
class LearningContextSelection:
    selected_agent_memories: list[dict[str, Any]] = field(default_factory=list)
    selected_user_memories: list[dict[str, Any]] = field(default_factory=list)
    selected_skill_summaries: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_agent_memories": self.selected_agent_memories,
            "selected_user_memories": self.selected_user_memories,
            "selected_skill_summaries": self.selected_skill_summaries,
            "budget": self.budget,
        }


def _tokens(text: str) -> set[str]:
    folded = str(text or "").casefold()
    return {part for part in folded.replace("_", " ").replace("-", " ").split() if part}


def _score_text(query: str, *parts: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    haystack = _tokens(" ".join(parts))
    overlap = len(query_tokens & haystack)
    return overlap / max(1, len(query_tokens))


def _trim(items: list[dict[str, Any]], *, budget: int, formatter) -> list[dict[str, Any]]:
    selected = []
    used = 0
    for item in items:
        text = formatter(item)
        if used + len(text) > budget and selected:
            break
        if len(text) > budget:
            item = {**item, "content": str(item.get("content") or text)[:budget]}
            text = formatter(item)
        selected.append(item)
        used += len(text)
    return selected


def select_agent_learning_context(
    *,
    session_id: str,
    message: str,
    agent_run_id: str | None = None,
    memory_budget: int = MEMORY_CHAR_BUDGET,
    skill_budget: int = SKILL_SUMMARY_CHAR_BUDGET,
) -> LearningContextSelection:
    memories = list_agent_memories(status="active", limit=100)
    legacy_user_read = os.getenv("MEMORY_LEGACY_USER_READ_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    scored_memories = []
    for item in memories:
        scope = str(item.get("scope") or "")
        if scope.startswith("user:") and contains_prohibited_sensitive_data(str(item.get("content") or "")):
            continue
        eligible = scope == "agent" or (legacy_user_read and scope == f"user:{session_id}")
        if not eligible:
            continue
        scope_boost = 0.2 if scope == "agent" else 0.05
        score = _score_text(message, item.get("content", ""), scope) + scope_boost + float(item.get("confidence") or 0.0) / 10
        scored_memories.append((score, item))
    scored_memories.sort(key=lambda pair: pair[0], reverse=True)
    selected_memories = _trim(
        [item for score, item in scored_memories if score > 0],
        budget=memory_budget,
        formatter=lambda item: f"{item.get('scope')}: {item.get('content')}\n",
    )

    skills = list_agent_skills(status="active", limit=100)
    scored_skills = []
    for item in skills:
        patterns = item.get("trigger_patterns") or []
        score = _score_text(
            message,
            item.get("name", ""),
            item.get("description", ""),
            " ".join(str(pattern) for pattern in patterns),
        )
        scored_skills.append((score, item))
    scored_skills.sort(key=lambda pair: pair[0], reverse=True)
    selected_skills = _trim(
        [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "description": item.get("description"),
                "risk_boundary": item.get("risk_boundary"),
                "version": item.get("version"),
                "trigger_patterns": item.get("trigger_patterns") or [],
            }
            for score, item in scored_skills
            if score > 0
        ],
        budget=skill_budget,
        formatter=lambda item: f"{item.get('name')}: {item.get('description')} Boundary: {item.get('risk_boundary')}\n",
    )

    for item in selected_memories:
        record_learning_usage(
            agent_run_id=agent_run_id,
            session_id=session_id,
            artifact_type="memory",
            artifact_id=int(item["id"]),
            usage_context="context_selection",
        )
    for item in selected_skills:
        record_learning_usage(
            agent_run_id=agent_run_id,
            session_id=session_id,
            artifact_type="skill",
            artifact_id=int(item["id"]),
            usage_context="skill_summary_selection",
        )

    return LearningContextSelection(
        selected_agent_memories=[item for item in selected_memories if item.get("scope") == "agent"],
        selected_user_memories=[item for item in selected_memories if item.get("scope") == f"user:{session_id}"],
        selected_skill_summaries=selected_skills,
        budget={"memory_chars": memory_budget, "skill_summary_chars": skill_budget},
    )


# Backward-compatible entry point for extensions. New runtime code uses the
# explicitly named agent-learning selector and retrieves user memory separately.
select_learning_context = select_agent_learning_context
