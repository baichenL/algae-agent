from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from app.core.db import connection
from app.core.time_utils import local_time_string, time_days_ago


ACTIVE_STATUSES = {"queued", "running", "waiting_input"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    try:
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
    except Exception:
        item["metadata"] = {}
    item["retryable"] = bool(item.get("retryable"))
    return item


def create_operation(
    *, owner: str, workspace_id: str, kind: str, label: str,
    phase: str = "queued", message: str | None = None,
    related_run_id: str | None = None, related_entity_type: str | None = None,
    related_entity_id: str | None = None, metadata: dict[str, Any] | None = None,
    related_task_id: str | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    operation_id = str(uuid.uuid4())
    now = local_time_string()
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            INSERT INTO background_operations
            (id, owner, workspace_id, kind, label, status, phase, message,
             related_run_id, related_entity_type, related_entity_id, retryable,
             metadata_json, related_task_id, created_at, heartbeat_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (operation_id, owner, workspace_id, kind, label, phase, message,
             related_run_id, related_entity_type, related_entity_id, 1 if retryable else 0,
             json.dumps(metadata or {}, ensure_ascii=False, default=str), related_task_id, now, now, now),
        )
        row = conn.execute("SELECT * FROM background_operations WHERE id = ?", (operation_id,)).fetchone()
        conn.commit()
    append_event(operation_id, "operation_created", phase=phase, message=message)
    return _decode(row) or {}


def get_operation(operation_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return _decode(conn.execute("SELECT * FROM background_operations WHERE id = ?", (operation_id,)).fetchone())


def find_latest_related(entity_type: str, entity_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM background_operations WHERE related_entity_type = ? AND related_entity_id = ? ORDER BY created_at DESC LIMIT 1",
            (entity_type, entity_id),
        ).fetchone()
    return _decode(row)


def fail_interrupted_operations() -> int:
    """Make interrupted non-approval work explicit instead of leaving stale spinners."""

    now = local_time_string()
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    with sqlite3.connect(connection.DB_PATH) as conn:
        cursor = conn.execute(
            f"""
            UPDATE background_operations
            SET status = 'failed', phase = 'interrupted',
                message = '服务重启，任务需要安全重试',
                error_code = 'SERVICE_RESTARTED',
                error_message = '后台任务在服务重启时中断',
                retryable = 1, completed_at = ?, heartbeat_at = ?, updated_at = ?
            WHERE status IN ({placeholders}) AND kind != 'approval_execution'
            """,
            (now, now, now, *sorted(ACTIVE_STATUSES)),
        )
        conn.commit()
        return cursor.rowcount


def list_operations(owner: str, workspace_id: str, *, scope: str = "active", limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if scope == "active":
            placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
            rows = conn.execute(
                f"SELECT * FROM background_operations WHERE owner = ? AND workspace_id = ? AND status IN ({placeholders}) ORDER BY updated_at DESC LIMIT ?",
                (owner, workspace_id, *sorted(ACTIVE_STATUSES), safe_limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM background_operations WHERE owner = ? AND workspace_id = ? AND updated_at >= ? ORDER BY updated_at DESC LIMIT ?",
                (owner, workspace_id, time_days_ago(1), safe_limit),
            ).fetchall()
    return [_decode(row) or {} for row in rows]


def update_operation(
    operation_id: str, *, status: str | None = None, phase: str | None = None,
    message: str | None = None, progress: float | None = None,
    related_run_id: str | None = None, error_code: str | None = None,
    error_message: str | None = None, retryable: bool | None = None,
    metadata: dict[str, Any] | None = None, related_task_id: str | None = None,
) -> dict[str, Any] | None:
    now = local_time_string()
    updates = ["heartbeat_at = ?", "updated_at = ?"]
    params: list[Any] = [now, now]
    values = {
        "status": status, "phase": phase, "message": message, "progress": progress,
        "related_run_id": related_run_id, "error_code": error_code,
        "error_message": error_message, "related_task_id": related_task_id,
    }
    for column, value in values.items():
        if value is not None:
            updates.append(f"{column} = ?")
            params.append(value)
    if retryable is not None:
        updates.append("retryable = ?")
        params.append(1 if retryable else 0)
    if metadata is not None:
        updates.append("metadata_json = ?")
        params.append(json.dumps(metadata, ensure_ascii=False, default=str))
    if status == "running":
        updates.append("started_at = COALESCE(started_at, ?)")
        params.append(now)
    if status in TERMINAL_STATUSES:
        updates.append("completed_at = ?")
        params.append(now)
    params.append(operation_id)
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(f"UPDATE background_operations SET {', '.join(updates)} WHERE id = ?", params)
        row = conn.execute("SELECT * FROM background_operations WHERE id = ?", (operation_id,)).fetchone()
        conn.commit()
    if row and (status or phase or message):
        append_event(operation_id, f"operation_{status or 'updated'}", phase=phase, message=message, payload={"progress": progress})
    return _decode(row)


def append_event(operation_id: str, event_type: str, *, phase: str | None = None, message: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    now = local_time_string()
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        sequence = int(conn.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM operation_events WHERE operation_id = ?", (operation_id,)).fetchone()[0])
        conn.execute(
            "INSERT INTO operation_events (operation_id, sequence, event_type, phase, message, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (operation_id, sequence, event_type, phase, message, json.dumps(payload or {}, ensure_ascii=False, default=str), now),
        )
        conn.commit()
    return {"sequence": sequence, "event_type": event_type, "phase": phase, "message": message, "payload": payload or {}, "created_at": now}


def list_events(operation_id: str, *, after: int = 0) -> list[dict[str, Any]]:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT sequence, event_type, phase, message, payload_json, created_at FROM operation_events WHERE operation_id = ? AND sequence > ? ORDER BY sequence",
            (operation_id, int(after)),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        result.append(item)
    return result
