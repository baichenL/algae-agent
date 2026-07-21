from __future__ import annotations

from app.core.db.connection import connect
from app.core.time_utils import days_since_time, local_time_string
from app.services.user_memory.store import record_memory_event


def curate_user_memories(owner_id: str, workspace_id: str) -> dict[str, int]:
    """Adjust confidence from explicit/repeated feedback; never auto-delete."""
    with connect(row_factory=True) as conn:
        rows = conn.execute(
            """
            SELECT m.*,
                   (SELECT COUNT(*) FROM user_memory_events e
                    WHERE e.memory_id = m.id AND e.event_type = 'noop'
                      AND e.actor = 'memory_extractor') AS repeat_count
            FROM user_memories m
            WHERE m.owner_id = ? AND m.workspace_id = ? AND m.status = 'active'
            """,
            (owner_id, workspace_id),
        ).fetchall()
    adjusted = 0
    for row in rows:
        item = dict(row)
        current = float(item["confidence"])
        if int(item["confirmation_count"] or 0) > 0:
            target = max(current, 0.95)
        else:
            repeat_boost = min(int(item["repeat_count"] or 0), 5) * 0.01
            correction_penalty = min(int(item["correction_count"] or 0), 5) * 0.05
            stale_periods = days_since_time(item.get("updated_at")) // 90
            target = current + repeat_boost - correction_penalty - min(stale_periods, 5) * 0.02
        target = round(min(max(target, 0.25), 1.0), 4)
        if target == current:
            continue
        with connect() as conn:
            conn.execute(
                "UPDATE user_memories SET confidence = ?, updated_at = ? WHERE id = ? AND owner_id = ? AND workspace_id = ?",
                (target, local_time_string(), item["id"], owner_id, workspace_id),
            )
            conn.commit()
        record_memory_event(
            owner_id=owner_id, workspace_id=workspace_id, memory_id=int(item["id"]),
            event_type="confidence_curated", actor="user_memory_curator",
            metadata={"previous_confidence": current, "new_confidence": target},
        )
        adjusted += 1
    return {"examined": len(rows), "adjusted": adjusted}


def curate_all_user_memories() -> dict[str, int]:
    with connect(row_factory=True) as conn:
        scopes = conn.execute(
            "SELECT DISTINCT owner_id, workspace_id FROM user_memories WHERE status = 'active'"
        ).fetchall()
    examined = adjusted = 0
    for scope in scopes:
        result = curate_user_memories(scope["owner_id"], scope["workspace_id"])
        examined += result["examined"]
        adjusted += result["adjusted"]
    return {"examined": examined, "adjusted": adjusted}
