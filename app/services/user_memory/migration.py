from __future__ import annotations

import os

from app.core.db.connection import connect
from app.services.user_memory.store import create_candidate, record_memory_event


def import_legacy_user_memories() -> dict[str, int]:
    if os.getenv("MEMORY_LEGACY_USER_READ_ENABLED", "true").strip().lower() not in {"1", "true", "yes"}:
        return {"imported": 0, "unresolved": 0}
    imported = 0
    unresolved = 0
    with connect(row_factory=True) as conn:
        rows = conn.execute("SELECT * FROM agent_memories WHERE scope LIKE 'user:%' ORDER BY id").fetchall()
    for row in rows:
        item = dict(row)
        conversation_id = str(item["scope"])[5:]
        with connect(row_factory=True) as conn:
            conversation = conn.execute(
                "SELECT owner, workspace_id FROM assistant_conversations WHERE id = ? OR legacy_session_id = ? LIMIT 1",
                (conversation_id, conversation_id),
            ).fetchone()
        if not conversation:
            unresolved += 1
            continue
        try:
            create_candidate(
                owner_id=conversation["owner"], workspace_id=conversation["workspace_id"], memory_type="note",
                predicate="general.note", value=item["content"], confidence=float(item.get("confidence") or 0.5),
                sensitivity="low", reason="legacy_user_memory_import", evidence="legacy memory",
                model_name="legacy_migration", source_conversation_id=conversation_id,
                source_message_id=-int(item["id"]), source_run_id=item.get("source_run_id"), status="pending",
            )
            imported += 1
        except ValueError:
            unresolved += 1
            record_memory_event(
                owner_id=conversation["owner"], workspace_id=conversation["workspace_id"],
                event_type="candidate_blocked", actor="legacy_memory_migration",
                reason="prohibited_or_invalid_legacy_memory",
                metadata={"legacy_memory_id": item["id"]},
            )
    return {"imported": imported, "unresolved": unresolved}
