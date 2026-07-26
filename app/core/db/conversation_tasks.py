from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from typing import Any

from app.core.db.connection import connect
from app.core.time_utils import local_time_string


ACTIVE_STATUSES = (
    "collecting",
    "ready",
    "waiting_approval",
    "running",
    "waiting_manual_confirmation",
)
TERMINAL_STATUSES = ("completed", "cancelled", "failed", "expired")


class TaskVersionConflict(RuntimeError):
    pass


class ActiveTaskConflict(RuntimeError):
    def __init__(self, task: dict[str, Any]):
        super().__init__("active_conversation_task_exists")
        self.task = task


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    item["state_changing"] = bool(item.get("state_changing"))
    for column, public_name, default in (
        ("collected_slots_json", "collected_slots", {}),
        ("missing_slots_json", "missing_slots", []),
        ("proposed_action_json", "proposed_action", None),
        ("task_spec_json", "task_spec", None),
    ):
        item[public_name] = _json_load(item.pop(column, None), default)
    return item


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _expiry(hours: int = 24) -> str:
    return (_utc_now() + dt.timedelta(hours=hours)).isoformat()


def append_task_event(task_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    with connect(row_factory=True) as conn:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        sequence = int(
            cursor.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM conversation_task_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0]
        )
        now = local_time_string()
        cursor.execute(
            """
            INSERT INTO conversation_task_events (task_id, sequence, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, sequence, event_type, _json_dump(payload or {}), now),
        )
        conn.commit()
    return {"task_id": task_id, "sequence": sequence, "event_type": event_type, "payload": payload or {}, "created_at": now}


def list_task_events(task_id: str) -> list[dict[str, Any]]:
    with connect(row_factory=True) as conn:
        rows = conn.execute(
            "SELECT * FROM conversation_task_events WHERE task_id = ? ORDER BY sequence",
            (task_id,),
        ).fetchall()
    return [
        {
            **dict(row),
            "payload": _json_load(row["payload_json"], {}),
        }
        for row in rows
    ]


def get_task(task_id: str) -> dict[str, Any] | None:
    with connect(row_factory=True) as conn:
        row = conn.execute("SELECT * FROM conversation_tasks WHERE id = ?", (task_id,)).fetchone()
    return _decode(row)


def get_task_by_legacy_source(legacy_source: str) -> dict[str, Any] | None:
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM conversation_tasks WHERE legacy_source = ?",
            (legacy_source,),
        ).fetchone()
    return _decode(row)


def get_latest_task(conversation_id: str) -> dict[str, Any] | None:
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM conversation_tasks WHERE conversation_id = ? ORDER BY updated_at DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
    return _decode(row)


def find_latest_task(
    conversation_id: str,
    *,
    task_type: str,
    owner: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any] | None:
    clauses = ["conversation_id = ?", "task_type = ?"]
    params: list[Any] = [conversation_id, task_type]
    if owner is not None:
        clauses.append("owner = ?")
        params.append(owner)
    if workspace_id is not None:
        clauses.append("workspace_id = ?")
        params.append(workspace_id)
    with connect(row_factory=True) as conn:
        row = conn.execute(
            f"SELECT * FROM conversation_tasks WHERE {' AND '.join(clauses)} "
            "ORDER BY updated_at DESC LIMIT 1",
            params,
        ).fetchone()
    return _decode(row)


def get_task_for_run(agent_run_id: str) -> dict[str, Any] | None:
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM conversation_tasks WHERE latest_agent_run_id = ? ORDER BY updated_at DESC LIMIT 1",
            (agent_run_id,),
        ).fetchone()
    return _decode(row)


def _expire_stale(conversation_id: str | None = None) -> None:
    now_iso = _utc_now().isoformat()
    params: list[Any] = [local_time_string(), local_time_string(), now_iso]
    where = ""
    if conversation_id:
        where = " AND conversation_id = ?"
        params.append(conversation_id)
    with connect() as conn:
        conn.execute(
            f"""
            UPDATE conversation_tasks
            SET status = 'expired', version = version + 1, updated_at = ?, completed_at = ?
            WHERE status IN ('collecting', 'ready')
              AND expires_at IS NOT NULL AND expires_at <= ?{where}
            """,
            params,
        )
        conn.commit()


def get_active_task(
    conversation_id: str,
    *,
    owner: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any] | None:
    _expire_stale(conversation_id)
    clauses = ["conversation_id = ?", "state_changing = 1", f"status IN ({','.join('?' for _ in ACTIVE_STATUSES)})"]
    params: list[Any] = [conversation_id, *ACTIVE_STATUSES]
    if owner is not None:
        clauses.append("owner = ?")
        params.append(owner)
    if workspace_id is not None:
        clauses.append("workspace_id = ?")
        params.append(workspace_id)
    with connect(row_factory=True) as conn:
        row = conn.execute(
            f"SELECT * FROM conversation_tasks WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT 1",
            params,
        ).fetchone()
    return _decode(row)


def list_tasks(
    conversation_id: str,
    *,
    owner: str,
    workspace_id: str,
    scope: str = "recent",
    limit: int = 30,
) -> list[dict[str, Any]]:
    _expire_stale(conversation_id)
    clauses = ["conversation_id = ?", "owner = ?", "workspace_id = ?"]
    params: list[Any] = [conversation_id, owner, workspace_id]
    if scope == "active":
        clauses.append(f"status IN ({','.join('?' for _ in ACTIVE_STATUSES)})")
        params.extend(ACTIVE_STATUSES)
    params.append(max(1, min(int(limit), 100)))
    with connect(row_factory=True) as conn:
        rows = conn.execute(
            f"SELECT * FROM conversation_tasks WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
            params,
        ).fetchall()
    return [_decode(row) for row in rows if row is not None]


def create_task(
    *,
    conversation_id: str,
    owner: str,
    workspace_id: str,
    goal_text: str,
    task_type: str = "conversation_turn",
    status: str = "running",
    state_changing: bool = False,
    collected_slots: dict[str, Any] | None = None,
    missing_slots: list[str] | None = None,
    proposed_action: dict[str, Any] | None = None,
    task_spec: dict[str, Any] | None = None,
    pending_id: int | None = None,
    workflow_run_id: str | None = None,
    latest_agent_run_id: str | None = None,
    parent_task_id: str | None = None,
    legacy_source: str | None = None,
    expires_at: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    task_id = task_id or str(uuid.uuid4())
    now = local_time_string()
    if status in {"collecting", "ready"} and expires_at is None:
        expires_at = _expiry()
    try:
        with connect(row_factory=True) as conn:
            conn.execute(
                """
                INSERT INTO conversation_tasks (
                    id, conversation_id, owner, workspace_id, task_type, goal_text, status,
                    state_changing, collected_slots_json, missing_slots_json, proposed_action_json,
                    task_spec_json, pending_id, workflow_run_id, latest_agent_run_id,
                    parent_task_id, version, legacy_source,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    conversation_id,
                    owner,
                    workspace_id,
                    task_type,
                    goal_text,
                    status,
                    int(state_changing),
                    _json_dump(collected_slots or {}),
                    _json_dump(missing_slots or []),
                    _json_dump(proposed_action) if proposed_action is not None else None,
                    _json_dump(task_spec) if task_spec is not None else None,
                    pending_id,
                    workflow_run_id,
                    latest_agent_run_id,
                    parent_task_id,
                    legacy_source,
                    now,
                    now,
                    expires_at,
                ),
            )
            row = conn.execute("SELECT * FROM conversation_tasks WHERE id = ?", (task_id,)).fetchone()
            conn.commit()
    except sqlite3.IntegrityError as exc:
        if legacy_source:
            with connect(row_factory=True) as conn:
                existing = conn.execute(
                    "SELECT * FROM conversation_tasks WHERE legacy_source = ?", (legacy_source,)
                ).fetchone()
            if existing:
                return _decode(existing) or {}
        active = get_active_task(conversation_id, owner=owner, workspace_id=workspace_id)
        if active:
            raise ActiveTaskConflict(active) from exc
        raise
    task = _decode(row) or {}
    append_task_event(task_id, "task_created", {"task_type": task_type, "status": status, "state_changing": state_changing})
    return task


def update_task(task_id: str, *, expected_version: int | None = None, event_type: str | None = None, **changes: Any) -> dict[str, Any]:
    allowed = {
        "task_type",
        "goal_text",
        "status",
        "state_changing",
        "pending_id",
        "workflow_run_id",
        "latest_agent_run_id",
        "expires_at",
        "completed_at",
        "parent_task_id",
        "superseded_by_task_id",
        "pause_reason",
    }
    json_columns = {
        "collected_slots": "collected_slots_json",
        "missing_slots": "missing_slots_json",
        "proposed_action": "proposed_action_json",
        "task_spec": "task_spec_json",
    }
    assignments: list[str] = []
    params: list[Any] = []
    for key, value in changes.items():
        if key in allowed:
            assignments.append(f"{key} = ?")
            params.append(int(value) if key == "state_changing" else value)
        elif key in json_columns:
            assignments.append(f"{json_columns[key]} = ?")
            params.append(_json_dump(value))
    if not assignments:
        task = get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        return task
    status = changes.get("status")
    if status in TERMINAL_STATUSES and "completed_at" not in changes:
        assignments.append("completed_at = ?")
        params.append(local_time_string())
    elif status in ACTIVE_STATUSES and "completed_at" not in changes:
        assignments.append("completed_at = NULL")
    if status in {"collecting", "ready"} and "expires_at" not in changes:
        assignments.append("expires_at = ?")
        params.append(_expiry())
    assignments.extend(["version = version + 1", "updated_at = ?"])
    params.append(local_time_string())
    where = "id = ?"
    params.append(task_id)
    if expected_version is not None:
        where += " AND version = ?"
        params.append(int(expected_version))
    with connect(row_factory=True) as conn:
        cursor = conn.execute(f"UPDATE conversation_tasks SET {', '.join(assignments)} WHERE {where}", params)
        if cursor.rowcount != 1:
            if get_task(task_id) is None:
                raise KeyError(task_id)
            raise TaskVersionConflict(task_id)
        row = conn.execute("SELECT * FROM conversation_tasks WHERE id = ?", (task_id,)).fetchone()
        conn.commit()
    task = _decode(row) or {}
    append_task_event(task_id, event_type or "task_updated", {key: task.get(key) for key in changes})
    return task


def cancel_task(task_id: str, *, reason: str = "cancelled_by_user") -> dict[str, Any]:
    task = get_task(task_id)
    if task is None:
        raise KeyError(task_id)
    if task["status"] in TERMINAL_STATUSES:
        return task
    return update_task(task_id, expected_version=int(task["version"]), status="cancelled", event_type="task_cancelled", proposed_action={**(task.get("proposed_action") or {}), "cancel_reason": reason})


def suspend_task(
    task_id: str,
    *,
    superseded_by_task_id: str | None = None,
) -> dict[str, Any]:
    task = get_task(task_id)
    if task is None:
        raise KeyError(task_id)
    if task["status"] in TERMINAL_STATUSES or task["status"] == "suspended":
        return task
    return update_task(
        task_id,
        expected_version=int(task["version"]),
        status="suspended",
        state_changing=False,
        superseded_by_task_id=superseded_by_task_id,
        event_type="task_suspended",
    )


def pause_task(
    task_id: str,
    *,
    reason: str,
    proposed_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = get_task(task_id)
    if task is None:
        raise KeyError(task_id)
    return update_task(
        task_id,
        expected_version=int(task["version"]),
        status="paused",
        state_changing=False,
        pause_reason=reason,
        proposed_action=(
            proposed_action if proposed_action is not None else task.get("proposed_action")
        ),
        event_type="task_paused",
    )


def graph_thread_id(task_id: str) -> str:
    return f"agent-task:{task_id}"
