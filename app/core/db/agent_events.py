from __future__ import annotations

import json
from typing import Any

from app.core.db.connection import connect
from app.core.time_utils import local_time_string


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, default=str)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def record_error_event(
    *,
    session_id: str | None = None,
    agent_run_id: str | None = None,
    layer: str,
    component: str,
    operation: str,
    severity: str = "error",
    error_type: str | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    """Persist a cross-layer error event.

    Error recording is intentionally best-effort: observability must not break
    the primary chat/tool/workflow path.
    """

    try:
        with connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_error_events (
                    session_id,
                    agent_run_id,
                    layer,
                    component,
                    operation,
                    severity,
                    error_type,
                    error_message,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    agent_run_id,
                    layer,
                    component,
                    operation,
                    severity,
                    error_type,
                    error_message,
                    _json_dumps(metadata),
                    local_time_string(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except Exception:
        return None


def start_agent_run(
    *,
    agent_run_id: str,
    session_id: str,
    user_message: str,
    graph_thread_id: str | None = None,
    graph_definition_version: str | None = None,
    state_schema_version: int | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    attempt_no: int = 1,
) -> str:
    try:
        with connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO agent_runs (
                    id,
                    session_id,
                    user_message,
                    status,
                    started_at,
                    graph_thread_id,
                    checkpoint_status,
                    graph_definition_version,
                    state_schema_version
                    , conversation_id, task_id, attempt_no
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_run_id,
                    session_id,
                    user_message,
                    "running",
                    local_time_string(),
                    graph_thread_id,
                    "running" if graph_thread_id else None,
                    graph_definition_version,
                    state_schema_version,
                    conversation_id or session_id,
                    task_id,
                    int(attempt_no),
                ),
            )
            conn.commit()
    except Exception as exc:
        record_error_event(
            session_id=session_id,
            agent_run_id=agent_run_id,
            layer="agent_runtime",
            component="agent_events",
            operation="start_agent_run",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return agent_run_id


def record_agent_run_event(
    *,
    agent_run_id: str | None,
    session_id: str | None = None,
    event_type: str,
    layer: str,
    payload: dict[str, Any] | None = None,
) -> int | None:
    if not agent_run_id:
        return None
    try:
        with connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_run_events (
                    agent_run_id,
                    session_id,
                    event_type,
                    layer,
                    payload_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_run_id,
                    session_id,
                    event_type,
                    layer,
                    _json_dumps(payload),
                    local_time_string(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
    except Exception as exc:
        record_error_event(
            session_id=session_id,
            agent_run_id=agent_run_id,
            layer="agent_runtime",
            component="agent_events",
            operation="record_agent_run_event",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"event_type": event_type},
        )
        return None


def finish_agent_run(
    *,
    agent_run_id: str | None,
    status: str,
    final_route: str | None = None,
    risk_level: str | None = None,
    response_summary: str | None = None,
    error_event_id: int | None = None,
) -> None:
    if not agent_run_id:
        return
    try:
        with connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE agent_runs
                SET
                    status = ?,
                    finished_at = ?,
                    final_route = COALESCE(?, final_route),
                    risk_level = COALESCE(?, risk_level),
                    response_summary = COALESCE(?, response_summary),
                    error_event_id = COALESCE(?, error_event_id)
                WHERE id = ?
                """,
                (
                    status,
                    local_time_string(),
                    final_route,
                    risk_level,
                    response_summary,
                    error_event_id,
                    agent_run_id,
                ),
            )
            conn.commit()
    except Exception as exc:
        record_error_event(
            agent_run_id=agent_run_id,
            layer="agent_runtime",
            component="agent_events",
            operation="finish_agent_run",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"status": status},
        )


def mark_agent_run_waiting_approval(
    *,
    agent_run_id: str,
    pending_id: int,
    graph_thread_id: str,
    last_node: str = "AwaitApproval",
) -> bool:
    try:
        with connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE agent_runs
                SET status = 'waiting_approval',
                    checkpoint_status = 'interrupted',
                    interrupted_at = COALESCE(interrupted_at, ?),
                    pending_id = ?,
                    graph_thread_id = COALESCE(graph_thread_id, ?),
                    last_node = ?
                WHERE id = ? AND status IN ('running', 'waiting_approval')
                """,
                (
                    local_time_string(),
                    pending_id,
                    graph_thread_id,
                    last_node,
                    agent_run_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
    except Exception as exc:
        record_error_event(
            agent_run_id=agent_run_id,
            layer="agent_runtime",
            component="agent_events",
            operation="mark_agent_run_waiting_approval",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"pending_id": pending_id, "graph_thread_id": graph_thread_id},
        )
        return False


def mark_agent_run_resumed(
    *,
    agent_run_id: str,
    graph_thread_id: str,
    last_node: str = "LoadApproval",
) -> None:
    try:
        with connect() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET status = 'running',
                    checkpoint_status = 'resumed',
                    resumed_at = ?,
                    resume_count = COALESCE(resume_count, 0) + 1,
                    graph_thread_id = COALESCE(graph_thread_id, ?),
                    last_node = ?
                WHERE id = ?
                """,
                (local_time_string(), graph_thread_id, last_node, agent_run_id),
            )
            conn.commit()
    except Exception as exc:
        record_error_event(
            agent_run_id=agent_run_id,
            layer="agent_runtime",
            component="agent_events",
            operation="mark_agent_run_resumed",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"graph_thread_id": graph_thread_id},
        )


def get_agent_run(agent_run_id: str) -> dict[str, Any] | None:
    with connect(row_factory=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agent_runs WHERE id = ?", (agent_run_id,))
        row = cursor.fetchone()
    return _row_to_dict(row) if row else None


def list_agent_runs(
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit or 20), 100))
    clauses = []
    params: list[Any] = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect(row_factory=True) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT *
            FROM agent_runs
            {where_sql}
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return [_row_to_dict(row) for row in cursor.fetchall()]


def list_agent_run_events(agent_run_id: str) -> list[dict[str, Any]]:
    with connect(row_factory=True) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM agent_run_events
            WHERE agent_run_id = ?
            ORDER BY id ASC
            """,
            (agent_run_id,),
        )
        rows = cursor.fetchall()
    events = []
    for row in rows:
        item = _row_to_dict(row)
        item["payload"] = _json_loads(item.get("payload_json"), {})
        events.append(item)
    return events


def list_agent_error_events(agent_run_id: str) -> list[dict[str, Any]]:
    with connect(row_factory=True) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM agent_error_events
            WHERE agent_run_id = ?
            ORDER BY id ASC
            """,
            (agent_run_id,),
        )
        rows = cursor.fetchall()
    errors = []
    for row in rows:
        item = _row_to_dict(row)
        item["metadata"] = _json_loads(item.get("metadata_json"), {})
        errors.append(item)
    return errors
