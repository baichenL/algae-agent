from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.core.db.agent_events import record_agent_run_event
from app.services.learning.store import (
    list_agent_memories,
    list_agent_skills,
    update_memory_status,
    update_skill_status,
)


def run_learning_curator(*, agent_run_id: str | None = None, session_id: str | None = None) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in list_agent_memories(limit=200):
        by_key[(str(item.get("scope")), str(item.get("content")))].append(item)
        attempts = int(item.get("success_count") or 0) + int(item.get("failure_count") or 0)
        if attempts >= 3 and int(item.get("failure_count") or 0) > int(item.get("success_count") or 0):
            update_memory_status(int(item["id"]), "stale")
            decisions.append({"artifact_type": "memory", "artifact_id": item["id"], "decision": "mark_stale_low_success"})

    for duplicates in by_key.values():
        active = [item for item in duplicates if item.get("status") == "active"]
        if len(active) <= 1:
            continue
        keep = max(active, key=lambda item: (float(item.get("confidence") or 0), int(item.get("success_count") or 0), -int(item["id"])))
        for item in active:
            if item["id"] != keep["id"]:
                update_memory_status(int(item["id"]), "archived")
                decisions.append({
                    "artifact_type": "memory",
                    "artifact_id": item["id"],
                    "decision": "archive_duplicate",
                    "kept_id": keep["id"],
                })

    for skill in list_agent_skills(limit=200):
        usage = int(skill.get("usage_count") or 0)
        success = int(skill.get("success_count") or 0)
        failure = int(skill.get("failure_count") or 0)
        if usage >= 5 and failure > success:
            update_skill_status(int(skill["id"]), "stale")
            decisions.append({"artifact_type": "skill", "artifact_id": skill["id"], "decision": "mark_stale_low_success"})

    if agent_run_id:
        record_agent_run_event(
            agent_run_id=agent_run_id,
            session_id=session_id,
            event_type="agent_learning_curator_completed",
            layer="agent_learning",
            payload={"decision_count": len(decisions), "decisions": decisions},
        )
    return {"status": "success", "decision_count": len(decisions), "decisions": decisions}
