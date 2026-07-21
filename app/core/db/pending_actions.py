import datetime
import json
import sqlite3
from typing import Any

from app.core.db.connection import DB_PATH
from app.core.time_utils import display_time_string, local_time_string


VALID_PENDING_STATUSES = {"pending", "approved", "denied"}


def _with_display_times(row: dict) -> dict:
    row["created_at"] = display_time_string(row.get("created_at"))
    row["reviewed_at"] = display_time_string(row.get("reviewed_at"))
    return row


def insert_pending_action(
    action_type: str,
    payload: dict,
    requester: str = "LLM",
    risk_level: str = "medium",
    source: str = "chat",
    agent_run_id: str | None = None,
    graph_thread_id: str | None = None,
    execution_idempotency_key: str | None = None,
    domain_dedupe_key: str | None = None,
    expires_at: str | None = None,
) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if execution_idempotency_key:
            row = conn.execute(
                """
                SELECT id FROM pending_actions
                WHERE execution_idempotency_key = ?
                LIMIT 1
                """,
                (execution_idempotency_key,),
            ).fetchone()
            if row:
                return int(row["id"])
        if domain_dedupe_key:
            row = conn.execute(
                """
                SELECT id FROM pending_actions
                WHERE domain_dedupe_key = ? AND status = 'pending'
                LIMIT 1
                """,
                (domain_dedupe_key,),
            ).fetchone()
            if row:
                return int(row["id"])
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO pending_actions
            (
                action_type,
                payload_json,
                created_at,
                requester,
                status,
                risk_level,
                source,
                agent_run_id,
                graph_thread_id,
                execution_idempotency_key,
                domain_dedupe_key,
                expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action_type,
                json.dumps(payload, ensure_ascii=False),
                local_time_string(),
                requester,
                "pending",
                risk_level,
                source,
                agent_run_id,
                graph_thread_id,
                execution_idempotency_key,
                domain_dedupe_key,
                expires_at,
            )
        )
        conn.commit()
        return cursor.lastrowid

def get_pending_action(action_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pending_actions WHERE id = ?", (action_id,))
        row = cursor.fetchone()
        if not row:
            return None
        r = dict(row)
        try:
            r["payload"] = json.loads(r.pop("payload_json") or "{}")
        except Exception:
            r["payload"] = {}
        try:
            r["execution_result"] = json.loads(r.get("execution_result_json") or "null")
        except Exception:
            r["execution_result"] = None
        return _with_display_times(r)

def list_pending_actions(status: str = "pending") -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if status == "all":
            cursor.execute("SELECT * FROM pending_actions ORDER BY id DESC")
        elif status in VALID_PENDING_STATUSES:
            cursor.execute("SELECT * FROM pending_actions WHERE status = ? ORDER BY id DESC", (status,))
        else:
            cursor.execute("SELECT * FROM pending_actions WHERE status = ? ORDER BY id DESC", ("pending",))
        rows = cursor.fetchall()
        out = []
        for row in rows:
            r = dict(row)
            try:
                r["payload"] = json.loads(r.pop("payload_json") or "{}")
            except Exception:
                r["payload"] = {}
            try:
                r["execution_result"] = json.loads(r.get("execution_result_json") or "null")
            except Exception:
                r["execution_result"] = None
            out.append(_with_display_times(r))
        return out

def update_pending_action_status(
    action_id: int,
    status: str,
    reviewed_by: str = "human",
    review_reason: str = None,
) -> bool:
    if status not in {"approved", "denied"}:
        return False

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE pending_actions
            SET status = ?, reviewed_at = ?, reviewed_by = ?, review_reason = ?
            WHERE id = ?
            """,
            (
                status,
                local_time_string(),
                reviewed_by,
                review_reason,
                action_id,
            )
        )
        conn.commit()
        return cursor.rowcount > 0


def review_pending_action_with_resume_job(
    action_id: int,
    *,
    approve: bool,
    reviewed_by: str = "human",
    review_reason: str | None = None,
) -> dict[str, Any]:
    decision = "approved" if approve else "denied"
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM pending_actions WHERE id = ?",
            (action_id,),
        ).fetchone()
        if not row:
            conn.rollback()
            return {"status": "error", "reason": "not_found"}
        pending = dict(row)
        if pending.get("status") not in {"pending", decision}:
            conn.rollback()
            return {"status": "error", "reason": "already_reviewed", "pending_status": pending.get("status")}
        current_version = int(pending.get("approval_version") or 0)
        if pending.get("status") == decision and current_version > 0:
            job = conn.execute(
                """
                SELECT * FROM approval_resume_jobs
                WHERE pending_id = ? AND approval_version = ?
                """,
                (action_id, current_version),
            ).fetchone()
            conn.commit()
            return {
                "status": "success",
                "idempotent": True,
                "pending_id": action_id,
                "decision": decision,
                "approval_version": current_version,
                "job_id": int(job["id"]) if job else None,
                "graph_thread_id": pending.get("graph_thread_id"),
                "agent_run_id": pending.get("agent_run_id"),
            }
        next_version = current_version + 1
        cursor = conn.execute(
            """
            UPDATE pending_actions
            SET status = ?, reviewed_at = ?, reviewed_by = ?, review_reason = ?,
                approval_version = ?, resume_requested_at = ?, resume_error = NULL
            WHERE id = ? AND status = 'pending'
            """,
            (
                decision,
                now,
                reviewed_by,
                review_reason,
                next_version,
                now,
                action_id,
            ),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            return {"status": "error", "reason": "cas_failed"}
        job_id = None
        if pending.get("graph_thread_id"):
            conn.execute(
                """
                INSERT OR IGNORE INTO approval_resume_jobs (
                    pending_id,
                    agent_run_id,
                    graph_thread_id,
                    approval_version,
                    decision,
                    status,
                    attempt_count,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?)
                """,
                (
                    action_id,
                    pending.get("agent_run_id"),
                    pending.get("graph_thread_id"),
                    next_version,
                    decision,
                    now,
                    now,
                ),
            )
            job = conn.execute(
                """
                SELECT * FROM approval_resume_jobs
                WHERE pending_id = ? AND approval_version = ?
                """,
                (action_id, next_version),
            ).fetchone()
            job_id = int(job["id"]) if job else None
        conn.commit()
        return {
            "status": "success",
            "pending_id": action_id,
            "decision": decision,
            "approval_version": next_version,
            "job_id": job_id,
            "graph_thread_id": pending.get("graph_thread_id"),
            "agent_run_id": pending.get("agent_run_id"),
        }


def claim_resume_job(job_id: int) -> dict[str, Any] | None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM approval_resume_jobs
            WHERE id = ? AND status IN ('pending', 'failed')
            """,
            (job_id,),
        ).fetchone()
        if not row:
            conn.rollback()
            return None
        conn.execute(
            """
            UPDATE approval_resume_jobs
            SET status = 'running', attempt_count = attempt_count + 1,
                last_error = NULL, updated_at = ?
            WHERE id = ?
            """,
            (now, job_id),
        )
        updated = conn.execute(
            "SELECT * FROM approval_resume_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        conn.commit()
        return dict(updated)


def mark_resume_job_succeeded(job_id: int) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE approval_resume_jobs
            SET status = 'succeeded', last_error = NULL, updated_at = ?
            WHERE id = ?
            """,
            (now, job_id),
        )
        conn.commit()


def mark_resume_job_failed(job_id: int, error: str) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE approval_resume_jobs
            SET status = 'failed', last_error = ?, updated_at = ?
            WHERE id = ?
            """,
            (error, now, job_id),
        )
        conn.commit()


def list_pending_resume_jobs(*, max_attempts: int = 5, limit: int = 20) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM approval_resume_jobs
            WHERE status IN ('pending', 'failed') AND attempt_count < ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (max_attempts, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def cleanup_completed_resume_jobs(*, older_than: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            DELETE FROM approval_resume_jobs
            WHERE status IN ('succeeded', 'failed') AND updated_at < ?
            """,
            (older_than,),
        )
        conn.commit()
        return cursor.rowcount


def mark_pending_resume_completed(action_id: int, result: dict[str, Any] | None = None) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE pending_actions
            SET resume_completed_at = ?, resume_error = NULL,
                execution_result_json = COALESCE(?, execution_result_json)
            WHERE id = ?
            """,
            (
                local_time_string(),
                json.dumps(result, ensure_ascii=False, default=str) if result is not None else None,
                action_id,
            ),
        )
        conn.commit()


def mark_pending_resume_failed(action_id: int, error: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE pending_actions
            SET resume_error = ?
            WHERE id = ?
            """,
            (error, action_id),
        )
        conn.commit()


def mark_pending_executed(action_id: int, result: dict[str, Any]) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE pending_actions
            SET executed_at = ?, execution_result_json = ?,
                resume_completed_at = COALESCE(resume_completed_at, ?),
                resume_error = NULL, execution_status = 'succeeded',
                execution_error = NULL
            WHERE id = ? AND executed_at IS NULL
            """,
            (
                local_time_string(),
                json.dumps(result, ensure_ascii=False, default=str),
                local_time_string(),
                action_id,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1


def queue_pending_execution(action_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE pending_actions
            SET execution_status = 'queued', execution_error = NULL
            WHERE id = ? AND status = 'approved' AND executed_at IS NULL
              AND COALESCE(execution_status, 'not_started') IN ('not_started', 'failed', 'queued')
            """,
            (action_id,),
        )
        conn.commit()
        return cursor.rowcount == 1


def claim_pending_execution(action_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE pending_actions
            SET execution_status = 'running', execution_started_at = ?, execution_error = NULL
            WHERE id = ? AND status = 'approved' AND executed_at IS NULL
              AND COALESCE(execution_status, 'not_started') IN ('not_started', 'queued', 'failed')
            """,
            (local_time_string(), action_id),
        )
        conn.commit()
        return cursor.rowcount == 1


def mark_pending_execution_failed(action_id: int, error: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE pending_actions
            SET execution_status = 'failed', execution_error = ?
            WHERE id = ? AND executed_at IS NULL
            """,
            (str(error)[:1000], action_id),
        )
        conn.commit()


def list_recoverable_pending_executions(limit: int = 100) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM pending_actions
            WHERE status = 'approved' AND executed_at IS NULL
              AND COALESCE(execution_status, 'not_started') IN ('not_started', 'queued', 'running', 'failed')
            ORDER BY id ASC LIMIT ?
            """,
            (max(1, min(int(limit), 500)),),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
        except Exception:
            item["payload"] = {}
        result.append(item)
    return result

def delete_pending_action(action_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pending_actions WHERE id = ?", (action_id,))
        conn.commit()
        return cursor.rowcount > 0

