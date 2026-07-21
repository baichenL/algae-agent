# app/core/db/workflow_runs.py
from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.core.db.connection import DB_PATH
from app.core.time_utils import local_time_string

# 作用是将数据库查询结果行解码为字典，并处理 JSON 字段和布尔字段
def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    raw_result = result.pop("tool_result_json", None)
    try:
        result["tool_result"] = json.loads(raw_result) if raw_result else None
    except json.JSONDecodeError:
        result["tool_result"] = None
    raw_hardware_state = result.pop("hardware_state_json", None)
    try:
        result["hardware_state"] = json.loads(raw_hardware_state) if raw_hardware_state else None
    except json.JSONDecodeError:
        result["hardware_state"] = None
    result["physical_execution"] = bool(result.get("physical_execution"))
    result["persisted"] = bool(result.get("persisted"))
    return result

# 作用是创建一个新的工作流运行记录
def create_workflow_run(
    *,
    pending_id: int,
    workflow_name: str,
    strain_id: str,
    execution_mode: str,
    expected_generation: int | None,
) -> dict[str, Any]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            INSERT OR IGNORE INTO workflow_runs
            (pending_id, workflow_name, strain_id, execution_mode, status,
             expected_generation, created_at)
            VALUES (?, ?, ?, ?, 'queued', ?, ?)
            """,
            (
                pending_id,
                workflow_name,
                strain_id,
                execution_mode,
                expected_generation,
                local_time_string(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM workflow_runs WHERE pending_id = ?",
            (pending_id,),
        ).fetchone()
        conn.commit()
        return _decode(row)


def approve_pending_and_create_run(
    *,
    pending_id: int,
    reviewed_by: str,
    workflow_name: str,
    strain_id: str,
    execution_mode: str,
    expected_generation: int | None,
) -> tuple[dict[str, Any], bool]:
    """Atomically approve one pending action and create its unique run."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        pending = conn.execute(
            "SELECT status FROM pending_actions WHERE id = ?",
            (pending_id,),
        ).fetchone()
        if not pending:
            conn.rollback()
            raise ValueError("not_found")
        existing = conn.execute(
            "SELECT * FROM workflow_runs WHERE pending_id = ?",
            (pending_id,),
        ).fetchone()
        if pending["status"] == "approved" and existing:
            conn.commit()
            return _decode(existing), True
        if pending["status"] not in {"pending", "approved"}:
            conn.rollback()
            raise ValueError("already_reviewed")
        if pending["status"] == "pending":
            conn.execute(
                """
                UPDATE pending_actions
                SET status = 'approved', reviewed_at = ?, reviewed_by = ?
                WHERE id = ? AND status = 'pending'
                """,
                (local_time_string(), reviewed_by, pending_id),
            )
        conn.execute(
            """
            INSERT INTO workflow_runs
            (pending_id, workflow_name, strain_id, execution_mode, status,
             expected_generation, created_at)
            VALUES (?, ?, ?, ?, 'queued', ?, ?)
            """,
            (
                pending_id,
                workflow_name,
                strain_id,
                execution_mode,
                expected_generation,
                local_time_string(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM workflow_runs WHERE pending_id = ?",
            (pending_id,),
        ).fetchone()
        conn.commit()
        return _decode(row), False


def get_workflow_run(run_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return _decode(conn.execute(
            "SELECT * FROM workflow_runs WHERE id = ?",
            (run_id,),
        ).fetchone())


def get_workflow_run_by_pending(pending_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return _decode(conn.execute(
            "SELECT * FROM workflow_runs WHERE pending_id = ?",
            (pending_id,),
        ).fetchone())


def list_workflow_runs(
    *,
    strain_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit or 20), 100))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if strain_id:
            rows = conn.execute(
                "SELECT * FROM workflow_runs WHERE strain_id = ? ORDER BY id DESC LIMIT ?",
                (strain_id, safe_limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM workflow_runs ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [_decode(row) for row in rows]


def claim_workflow_run(run_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE workflow_runs
            SET status = 'running', started_at = ?, updated_at = ?
            WHERE id = ? AND status = 'queued'
            """,
            (local_time_string(), local_time_string(), run_id),
        )
        conn.commit()
        return cursor.rowcount == 1


def update_workflow_simulation_state(
    run_id: int,
    *,
    current_step: str | None,
    progress: float,
    hardware_state: dict[str, Any] | None,
    event_sequence: int,
    event_at: str,
) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE workflow_runs
            SET current_step = ?, progress = ?, hardware_state_json = ?,
                last_event_sequence = ?, last_event_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                current_step,
                max(0.0, min(float(progress), 1.0)),
                json.dumps(hardware_state or {}, ensure_ascii=False, default=str),
                int(event_sequence),
                event_at,
                local_time_string(),
                run_id,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1


def requeue_interrupted_workflow_run(run_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE workflow_runs
            SET status = 'queued', started_at = NULL, error = 'recovered_after_restart'
            WHERE id = ? AND status = 'running' AND persisted = 0
            """,
            (run_id,),
        )
        conn.commit()
        return cursor.rowcount == 1


def finish_workflow_run(
    run_id: int,
    *,
    status: str,
    tool_result: dict[str, Any] | None,
    physical_execution: bool,
    persisted: bool,
    error: str | None = None,
) -> bool:
    if status not in {"succeeded", "failed", "cancelled", "stale"}:
        raise ValueError("invalid_workflow_run_status")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE workflow_runs
            SET status = ?, tool_result_json = ?, physical_execution = ?,
                persisted = ?, error = ?, finished_at = ?, updated_at = ?,
                progress = CASE WHEN ? = 'succeeded' THEN 1.0 ELSE progress END
            WHERE id = ? AND status = 'running'
            """,
            (
                status,
                json.dumps(tool_result, ensure_ascii=False) if tool_result is not None else None,
                int(physical_execution),
                int(persisted),
                error,
                local_time_string(),
                local_time_string(),
                status,
                run_id,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1


def commit_manual_subculture(
    run_id: int,
    *,
    completed_at: str,
    confirmed_by: str,
    note: str | None,
    idempotency_key: str,
) -> dict[str, Any]:
    """Atomically attest a manual subculture and persist the laboratory fact."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM workflow_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            conn.rollback()
            return {"status": "error", "reason": "run_not_found"}
        run = _decode(row)
        if run.get("persisted"):
            conn.commit()
            return {"status": "success", "idempotent": True, "run": run}
        if run.get("status") != "succeeded":
            conn.rollback()
            return {"status": "error", "reason": "workflow_not_succeeded"}
        if run.get("physical_execution"):
            conn.rollback()
            return {"status": "error", "reason": "physical_result_already_committed"}

        strain = conn.execute(
            "SELECT * FROM algae_status WHERE strain_id = ?", (run["strain_id"],)
        ).fetchone()
        if not strain:
            conn.rollback()
            return {"status": "error", "reason": "strain_not_found"}
        current_generation = int(strain["generation_number"] or 0)
        expected_generation = run.get("expected_generation")
        if expected_generation is not None and current_generation != int(expected_generation):
            conn.rollback()
            return {
                "status": "error",
                "reason": "stale_generation",
                "expected_generation": int(expected_generation),
                "current_generation": current_generation,
            }

        next_generation = current_generation + 1
        tool_result = dict(run.get("tool_result") or {})
        simulated_generation = tool_result.get("current_generation")
        if simulated_generation is not None and int(simulated_generation) != next_generation:
            conn.rollback()
            return {"status": "error", "reason": "simulation_generation_mismatch"}
        tool_result.update({
            "persisted": True,
            "commit_source": "manual_confirmation",
            "manual_confirmed_at": completed_at,
            "manual_confirmed_by": confirmed_by,
            "manual_note": note,
            "current_generation": next_generation,
            "days_counter": 0,
            "inoculation_time": completed_at,
        })
        now = local_time_string()
        conn.execute(
            """
            UPDATE algae_status
            SET generation_number = ?, days_since_last_subculture = 0,
                last_subculture_time = ?
            WHERE strain_id = ?
            """,
            (next_generation, completed_at, run["strain_id"]),
        )
        conn.execute(
            """
            INSERT INTO experiments
            (strain, generation, status, hardware_logs, created_at)
            VALUES (?, ?, 'SUCCESS_MANUAL_CONFIRMED', ?, ?)
            """,
            (
                run["strain_id"],
                next_generation,
                json.dumps(tool_result.get("execution_logs") or [], ensure_ascii=False),
                completed_at,
            ),
        )
        conn.execute(
            """
            INSERT INTO algae_audit (strain_id, action, details_json, performed_at)
            VALUES (?, 'manual_subculture_confirmed', ?, ?)
            """,
            (
                run["strain_id"],
                json.dumps({
                    "workflow_run_id": run_id,
                    "generation_number": next_generation,
                    "last_subculture_time": completed_at,
                    "confirmed_by": confirmed_by,
                    "note": note,
                }, ensure_ascii=False),
                now,
            ),
        )
        conn.execute(
            """
            UPDATE workflow_runs
            SET persisted = 1, tool_result_json = ?, commit_source = 'manual_confirmation',
                manual_confirmed_at = ?, manual_confirmed_by = ?, manual_note = ?,
                manual_commit_key = ?
            WHERE id = ? AND persisted = 0
            """,
            (
                json.dumps(tool_result, ensure_ascii=False),
                completed_at,
                confirmed_by,
                note,
                idempotency_key,
                run_id,
            ),
        )
        updated = conn.execute("SELECT * FROM workflow_runs WHERE id = ?", (run_id,)).fetchone()
        conn.commit()
        return {"status": "success", "idempotent": False, "run": _decode(updated)}
