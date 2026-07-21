from __future__ import annotations

import json
from typing import Any

from app.core.db.connection import connect
from app.core.time_utils import local_time_string
from app.services.user_memory.catalog import (
    PREDICATES,
    canonical_json,
    normalize_value,
    searchable_text,
    validate_memory_input,
    value_hash,
)


class MemoryRevisionConflict(RuntimeError):
    pass


def _decode(row: Any | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    for key in ("value_json", "metadata_json"):
        if key in item:
            raw = item.pop(key)
            public = key.removesuffix("_json")
            try:
                item[public] = json.loads(raw) if raw else None
            except Exception:
                item[public] = None
    for key in ("enabled", "auto_write_low_risk"):
        if key in item:
            item[key] = bool(item[key])
    return item


def _event(conn, *, owner_id: str, workspace_id: str, event_type: str, memory_id: int | None = None,
           candidate_id: int | None = None, actor: str | None = None, reason: str | None = None,
           metadata: dict[str, Any] | None = None) -> None:
    conn.execute(
        """
        INSERT INTO user_memory_events
        (memory_id, candidate_id, owner_id, workspace_id, event_type, actor, reason, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, candidate_id, owner_id, workspace_id, event_type, actor, reason,
         canonical_json(metadata or {}), local_time_string()),
    )


def record_memory_event(*, owner_id: str, workspace_id: str, event_type: str,
                        actor: str = "system", reason: str | None = None,
                        memory_id: int | None = None, candidate_id: int | None = None,
                        metadata: dict[str, Any] | None = None) -> None:
    """Record a content-free operational event for metrics and audit."""
    with connect() as conn:
        _event(
            conn, owner_id=owner_id, workspace_id=workspace_id, event_type=event_type,
            memory_id=memory_id, candidate_id=candidate_id, actor=actor,
            reason=reason, metadata=metadata,
        )
        conn.commit()


def get_settings(owner_id: str, workspace_id: str) -> dict[str, Any]:
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM user_memory_settings WHERE owner_id = ? AND workspace_id = ?",
            (owner_id, workspace_id),
        ).fetchone()
    if row:
        return _decode(row) or {}
    return {"owner_id": owner_id, "workspace_id": workspace_id, "enabled": False, "auto_write_low_risk": True}


def update_settings(owner_id: str, workspace_id: str, *, enabled: bool, auto_write_low_risk: bool) -> dict[str, Any]:
    now = local_time_string()
    with connect(row_factory=True) as conn:
        conn.execute(
            """
            INSERT INTO user_memory_settings
            (owner_id, workspace_id, enabled, auto_write_low_risk, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id, workspace_id) DO UPDATE SET
                enabled = excluded.enabled,
                auto_write_low_risk = excluded.auto_write_low_risk,
                updated_at = excluded.updated_at
            """,
            (owner_id, workspace_id, int(enabled), int(auto_write_low_risk), now, now),
        )
        row = conn.execute(
            "SELECT * FROM user_memory_settings WHERE owner_id = ? AND workspace_id = ?",
            (owner_id, workspace_id),
        ).fetchone()
        conn.commit()
    return _decode(row) or {}


def get_memory(memory_id: int, *, owner_id: str, workspace_id: str) -> dict[str, Any] | None:
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM user_memories WHERE id = ? AND owner_id = ? AND workspace_id = ?",
            (memory_id, owner_id, workspace_id),
        ).fetchone()
    return _decode(row)


def list_memories(*, owner_id: str, workspace_id: str, status: str | None = None,
                  memory_type: str | None = None, limit: int = 50, after_id: int | None = None) -> list[dict[str, Any]]:
    clauses = ["owner_id = ?", "workspace_id = ?"]
    params: list[Any] = [owner_id, workspace_id]
    if status:
        clauses.append("status = ?")
        params.append(status)
    if memory_type:
        clauses.append("memory_type = ?")
        params.append(memory_type)
    if after_id:
        clauses.append("id < ?")
        params.append(int(after_id))
    params.append(max(1, min(int(limit), 200)))
    with connect(row_factory=True) as conn:
        rows = conn.execute(
            f"SELECT * FROM user_memories WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
    return [_decode(row) or {} for row in rows]


def _sync_fts(conn, memory: dict[str, Any]) -> None:
    conn.execute("DELETE FROM user_memories_fts WHERE memory_id = ?", (memory["id"],))
    if memory.get("status") == "active":
        conn.execute(
            "INSERT INTO user_memories_fts (memory_id, owner_id, workspace_id, predicate, search_text) VALUES (?, ?, ?, ?, ?)",
            (memory["id"], memory["owner_id"], memory["workspace_id"], memory["predicate"], memory["search_text"]),
        )


def apply_memory(*, owner_id: str, workspace_id: str, memory_type: str, predicate: str, value: Any,
                 subject: str = "user", confidence: float = 1.0, sensitivity: str = "low",
                 source_conversation_id: str | None = None, source_message_id: int | None = None,
                 source_run_id: str | None = None, actor: str = "system", reason: str | None = None,
                 confirmation: bool = False) -> tuple[str, dict[str, Any]]:
    definition = validate_memory_input(memory_type, predicate, value)
    normalized = normalize_value(value)
    now = local_time_string()
    with connect(row_factory=True) as conn:
        existing = conn.execute(
            """
            SELECT * FROM user_memories
            WHERE owner_id = ? AND workspace_id = ? AND subject = ? AND predicate = ? AND status = 'active'
            ORDER BY revision DESC, id DESC LIMIT 1
            """,
            (owner_id, workspace_id, subject, predicate),
        ).fetchone()
        existing_item = _decode(existing)
        max_revision = int(conn.execute(
            """
            SELECT COALESCE(MAX(revision), 0) FROM user_memories
            WHERE owner_id = ? AND workspace_id = ? AND subject = ? AND predicate = ?
            """,
            (owner_id, workspace_id, subject, predicate),
        ).fetchone()[0])
        if existing_item and existing_item["normalized_value"] == normalized:
            conn.execute(
                """
                UPDATE user_memories
                SET confidence = MAX(confidence, ?), confirmation_count = confirmation_count + ?, updated_at = ?
                WHERE id = ?
                """,
                (float(confidence), int(confirmation), now, existing_item["id"]),
            )
            row = conn.execute("SELECT * FROM user_memories WHERE id = ?", (existing_item["id"],)).fetchone()
            _event(conn, memory_id=existing_item["id"], owner_id=owner_id, workspace_id=workspace_id,
                   event_type="confirmed" if confirmation else "noop", actor=actor, reason=reason)
            conn.commit()
            return "NOOP", _decode(row) or {}

        revision = max_revision + 1
        if existing_item and definition.singleton:
            conn.execute(
                "UPDATE user_memories SET status = 'superseded', valid_to = ?, correction_count = correction_count + 1, updated_at = ? WHERE id = ?",
                (now, now, existing_item["id"]),
            )
            conn.execute("DELETE FROM user_memories_fts WHERE memory_id = ?", (existing_item["id"],))
            _event(conn, memory_id=existing_item["id"], owner_id=owner_id, workspace_id=workspace_id,
                   event_type="superseded", actor=actor, reason=reason, metadata={"replacement_revision": revision})

        cursor = conn.execute(
            """
            INSERT INTO user_memories
            (owner_id, workspace_id, memory_type, subject, predicate, value_json, normalized_value,
             search_text, confidence, sensitivity, status, revision, valid_from, source_conversation_id,
             source_message_id, source_run_id, confirmation_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (owner_id, workspace_id, memory_type, subject, predicate, canonical_json(value), normalized,
             searchable_text(predicate, value), min(max(float(confidence), 0.0), 1.0), sensitivity, revision,
             now, source_conversation_id, source_message_id, source_run_id, int(confirmation), now, now),
        )
        row = conn.execute("SELECT * FROM user_memories WHERE id = ?", (cursor.lastrowid,)).fetchone()
        item = _decode(row) or {}
        _sync_fts(conn, item)
        action = "UPDATE" if existing_item and definition.singleton else "ADD"
        _event(conn, memory_id=item["id"], owner_id=owner_id, workspace_id=workspace_id,
               event_type="updated" if action == "UPDATE" else "created", actor=actor, reason=reason,
               metadata={"predicate": predicate, "revision": revision})
        conn.commit()
    return action, item


def update_memory(memory_id: int, *, owner_id: str, workspace_id: str, expected_revision: int,
                  value: Any, actor: str) -> dict[str, Any]:
    current = get_memory(memory_id, owner_id=owner_id, workspace_id=workspace_id)
    if current is None:
        raise KeyError(memory_id)
    if int(current["revision"]) != int(expected_revision) or current["status"] != "active":
        raise MemoryRevisionConflict(memory_id)
    _, updated = apply_memory(
        owner_id=owner_id, workspace_id=workspace_id, memory_type=current["memory_type"],
        predicate=current["predicate"], subject=current["subject"], value=value,
        confidence=1.0, sensitivity=current["sensitivity"], actor=actor,
        reason="user_correction", confirmation=True,
    )
    return updated


def archive_memory(memory_id: int, *, owner_id: str, workspace_id: str, actor: str) -> dict[str, Any]:
    now = local_time_string()
    with connect(row_factory=True) as conn:
        cursor = conn.execute(
            "UPDATE user_memories SET status = 'archived', valid_to = ?, updated_at = ? WHERE id = ? AND owner_id = ? AND workspace_id = ?",
            (now, now, memory_id, owner_id, workspace_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(memory_id)
        conn.execute("DELETE FROM user_memories_fts WHERE memory_id = ?", (memory_id,))
        _event(conn, memory_id=memory_id, owner_id=owner_id, workspace_id=workspace_id, event_type="archived", actor=actor)
        row = conn.execute("SELECT * FROM user_memories WHERE id = ?", (memory_id,)).fetchone()
        conn.commit()
    return _decode(row) or {}


def restore_memory(memory_id: int, *, owner_id: str, workspace_id: str, actor: str) -> dict[str, Any]:
    current = get_memory(memory_id, owner_id=owner_id, workspace_id=workspace_id)
    if current is None:
        raise KeyError(memory_id)
    if current["status"] == "active":
        return current
    _, restored = apply_memory(
        owner_id=owner_id, workspace_id=workspace_id, memory_type=current["memory_type"],
        predicate=current["predicate"], value=current["value"], subject=current["subject"],
        confidence=current["confidence"], sensitivity=current["sensitivity"], actor=actor,
        reason="restore", confirmation=True,
    )
    with connect() as conn:
        conn.execute(
            "UPDATE user_memories SET status = 'superseded', updated_at = ? WHERE id = ? AND owner_id = ? AND workspace_id = ?",
            (local_time_string(), memory_id, owner_id, workspace_id),
        )
        _event(
            conn, memory_id=memory_id, owner_id=owner_id, workspace_id=workspace_id,
            event_type="restored", actor=actor,
            metadata={"restored_as_memory_id": restored["id"], "revision": restored["revision"]},
        )
        conn.commit()
    return restored


def delete_memory(memory_id: int, *, owner_id: str, workspace_id: str, actor: str) -> None:
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT id FROM user_memories WHERE id = ? AND owner_id = ? AND workspace_id = ?",
            (memory_id, owner_id, workspace_id),
        ).fetchone()
        if not row:
            raise KeyError(memory_id)
        conn.execute("DELETE FROM user_memories_fts WHERE memory_id = ?", (memory_id,))
        conn.execute("DELETE FROM user_memory_embeddings WHERE memory_id = ?", (memory_id,))
        conn.execute("DELETE FROM user_memory_candidates WHERE applied_memory_id = ?", (memory_id,))
        conn.execute("DELETE FROM user_memories WHERE id = ?", (memory_id,))
        _event(conn, memory_id=memory_id, owner_id=owner_id, workspace_id=workspace_id,
               event_type="deleted", actor=actor, reason="privacy_delete")
        conn.commit()


def create_candidate(*, owner_id: str, workspace_id: str, memory_type: str, predicate: str, value: Any,
                     confidence: float, sensitivity: str, reason: str, evidence: str,
                     model_name: str | None, source_conversation_id: str | None,
                     source_message_id: int | None, source_run_id: str | None,
                     status: str = "pending") -> dict[str, Any]:
    validate_memory_input(memory_type, predicate, value, automatic=False)
    normalized = normalize_value(value)
    now = local_time_string()
    with connect(row_factory=True) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO user_memory_candidates
            (owner_id, workspace_id, memory_type, subject, predicate, value_json, normalized_value,
             normalized_hash, confidence, sensitivity, reason, evidence, model_name, status,
             source_conversation_id, source_message_id, source_run_id, created_at, updated_at)
            VALUES (?, ?, ?, 'user', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (owner_id, workspace_id, memory_type, predicate, canonical_json(value), normalized,
             value_hash(value), float(confidence), sensitivity, reason, evidence[:300], model_name, status,
             source_conversation_id, source_message_id, source_run_id, now, now),
        )
        row = conn.execute(
            """
            SELECT * FROM user_memory_candidates
            WHERE owner_id = ? AND workspace_id = ? AND source_message_id IS ? AND predicate = ? AND normalized_hash = ?
            """,
            (owner_id, workspace_id, source_message_id, predicate, value_hash(value)),
        ).fetchone()
        item = _decode(row) or {}
        _event(conn, candidate_id=item.get("id"), owner_id=owner_id, workspace_id=workspace_id,
               event_type="candidate_extracted" if status != "blocked" else "candidate_blocked",
               actor="memory_extractor", reason=reason, metadata={"predicate": predicate, "status": status})
        conn.commit()
    return item


def list_candidates(*, owner_id: str, workspace_id: str, status: str = "pending", limit: int = 100) -> list[dict[str, Any]]:
    with connect(row_factory=True) as conn:
        rows = conn.execute(
            """
            SELECT * FROM user_memory_candidates
            WHERE owner_id = ? AND workspace_id = ? AND status = ?
            ORDER BY id DESC LIMIT ?
            """,
            (owner_id, workspace_id, status, max(1, min(int(limit), 200))),
        ).fetchall()
    return [_decode(row) or {} for row in rows]


def decide_candidate(candidate_id: int, *, owner_id: str, workspace_id: str, accept: bool, actor: str) -> dict[str, Any]:
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM user_memory_candidates WHERE id = ? AND owner_id = ? AND workspace_id = ?",
            (candidate_id, owner_id, workspace_id),
        ).fetchone()
    candidate = _decode(row)
    if candidate is None:
        raise KeyError(candidate_id)
    if candidate["status"] != "pending":
        return candidate
    memory_id = None
    if accept:
        _, memory = apply_memory(
            owner_id=owner_id, workspace_id=workspace_id, memory_type=candidate["memory_type"],
            predicate=candidate["predicate"], value=candidate["value"], confidence=candidate["confidence"],
            sensitivity=candidate["sensitivity"], source_conversation_id=candidate.get("source_conversation_id"),
            source_message_id=candidate.get("source_message_id"), source_run_id=candidate.get("source_run_id"),
            actor=actor, reason="candidate_accepted", confirmation=True,
        )
        memory_id = memory["id"]
    now = local_time_string()
    with connect(row_factory=True) as conn:
        conn.execute(
            "UPDATE user_memory_candidates SET status = ?, applied_memory_id = ?, updated_at = ? WHERE id = ?",
            ("accepted" if accept else "rejected", memory_id, now, candidate_id),
        )
        _event(conn, candidate_id=candidate_id, memory_id=memory_id, owner_id=owner_id, workspace_id=workspace_id,
               event_type="candidate_accepted" if accept else "candidate_rejected", actor=actor)
        row = conn.execute("SELECT * FROM user_memory_candidates WHERE id = ?", (candidate_id,)).fetchone()
        conn.commit()
    return _decode(row) or {}


def increment_usage(memory_ids: list[int], *, owner_id: str, workspace_id: str, query: str) -> None:
    if not memory_ids:
        return
    now = local_time_string()
    with connect() as conn:
        for memory_id in memory_ids:
            conn.execute(
                "UPDATE user_memories SET use_count = use_count + 1, last_used_at = ?, updated_at = ? WHERE id = ? AND owner_id = ? AND workspace_id = ?",
                (now, now, memory_id, owner_id, workspace_id),
            )
            _event(conn, memory_id=memory_id, owner_id=owner_id, workspace_id=workspace_id,
                   event_type="retrieved", actor="context_retriever", metadata={"query_hash": value_hash(query)})
        conn.commit()
