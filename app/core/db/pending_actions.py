import datetime
import json
import os
import sqlite3
from typing import Any

from app.core.db.connection import DB_PATH
from app.core.time_utils import display_time_string, local_now, local_time_string, parse_time
from app.core.effects import stable_hash


VALID_PENDING_STATUSES = {"pending", "approved", "denied", "expired", "stale"}
DEFAULT_PENDING_TTL_SECONDS = max(60, int(os.getenv("PENDING_DEFAULT_TTL_SECONDS", "86400")))


def _default_expiry() -> str:
    return local_time_string(local_now() + datetime.timedelta(seconds=DEFAULT_PENDING_TTL_SECONDS))


def _is_expired(expires_at: str | None) -> bool:
    parsed = parse_time(expires_at)
    return bool(parsed and parsed <= local_now())


def _expire_locked(conn: sqlite3.Connection, action_id: int, expires_at: str | None) -> bool:
    if not _is_expired(expires_at):
        return False
    cursor = conn.execute(
        """
        UPDATE pending_actions
        SET status = 'expired', execution_status = 'expired',
            execution_error = COALESCE(execution_error, 'pending_expired')
        WHERE id = ? AND status IN ('pending', 'approved') AND executed_at IS NULL
        """,
        (action_id,),
    )
    return cursor.rowcount == 1


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
    proposal_version: int = 1,
    proposal_hash: str | None = None,
    proposal_envelope: dict[str, Any] | None = None,
    policy_version: str | None = None,
    expected_resource_versions: dict[str, Any] | None = None,
    created_by_tool_call_id: str | None = None,
) -> int:
    expires_at = expires_at or _default_expiry()
    proposal_envelope = dict(proposal_envelope or {})
    proposal_hash = proposal_hash or (
        stable_hash(proposal_envelope) if proposal_envelope else None
    )
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
                expires_at,
                proposal_version,
                proposal_hash,
                proposal_envelope_json,
                policy_version,
                expected_resource_versions_json,
                created_by_tool_call_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                int(proposal_version),
                proposal_hash,
                json.dumps(proposal_envelope, ensure_ascii=False, default=str)
                if proposal_envelope
                else None,
                policy_version,
                json.dumps(expected_resource_versions or {}, ensure_ascii=False, default=str),
                created_by_tool_call_id,
            )
        )
        conn.commit()
        return cursor.lastrowid

def get_pending_action(action_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pending_actions WHERE id = ?", (action_id,))
        row = cursor.fetchone()
        if not row:
            conn.rollback()
            return None
        if _expire_locked(conn, action_id, row["expires_at"]):
            row = conn.execute("SELECT * FROM pending_actions WHERE id = ?", (action_id,)).fetchone()
        conn.commit()
        r = dict(row)
        try:
            r["payload"] = json.loads(r.pop("payload_json") or "{}")
        except Exception:
            r["payload"] = {}
        try:
            r["execution_result"] = json.loads(r.get("execution_result_json") or "null")
        except Exception:
            r["execution_result"] = None
        for source, target, fallback in (
            ("proposal_envelope_json", "proposal_envelope", {}),
            ("expected_resource_versions_json", "expected_resource_versions", {}),
        ):
            try:
                r[target] = json.loads(r.get(source) or "null") or fallback
            except Exception:
                r[target] = fallback
        return _with_display_times(r)

def list_pending_actions(status: str = "pending") -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        expired_rows = conn.execute(
            """
            SELECT id, expires_at FROM pending_actions
            WHERE status IN ('pending', 'approved') AND executed_at IS NULL
              AND expires_at IS NOT NULL
            """
        ).fetchall()
        for expired_row in expired_rows:
            _expire_locked(conn, int(expired_row["id"]), expired_row["expires_at"])
        conn.commit()
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
    result = review_pending_action_with_resume_job(
        action_id,
        approve=status == "approved",
        reviewed_by=reviewed_by,
        review_reason=review_reason,
    )
    return result.get("status") == "success"


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
        if _expire_locked(conn, action_id, pending.get("expires_at")):
            conn.commit()
            return {"status": "error", "reason": "expired", "pending_status": "expired"}
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
            SELECT j.*, p.expires_at AS pending_expires_at,
                   p.status AS pending_status
            FROM approval_resume_jobs j
            JOIN pending_actions p ON p.id = j.pending_id
            WHERE j.id = ? AND j.status IN ('pending', 'failed')
            """,
            (job_id,),
        ).fetchone()
        if not row:
            conn.rollback()
            return None
        if _expire_locked(conn, int(row["pending_id"]), row["pending_expires_at"]):
            conn.execute(
                """
                UPDATE approval_resume_jobs
                SET status = 'failed', last_error = 'pending_expired', updated_at = ?
                WHERE id = ?
                """,
                (now, job_id),
            )
            conn.commit()
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
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT expires_at FROM pending_actions WHERE id = ?",
            (action_id,),
        ).fetchone()
        if not row or _expire_locked(conn, action_id, row["expires_at"]):
            conn.commit()
            return False
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


def mark_pending_stale(action_id: int, reason: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE pending_actions
            SET status = 'stale', execution_status = 'stale', execution_error = ?
            WHERE id = ? AND status = 'approved' AND executed_at IS NULL
            """,
            (reason, action_id),
        )
        conn.commit()
        return cursor.rowcount == 1


def queue_pending_execution(action_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT expires_at FROM pending_actions WHERE id = ?",
            (action_id,),
        ).fetchone()
        if not row or _expire_locked(conn, action_id, row["expires_at"]):
            conn.commit()
            return False
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
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT expires_at FROM pending_actions WHERE id = ?",
            (action_id,),
        ).fetchone()
        if not row or _expire_locked(conn, action_id, row["expires_at"]):
            conn.commit()
            return False
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
        conn.execute("BEGIN IMMEDIATE")
        candidates = conn.execute(
            """
            SELECT id, expires_at FROM pending_actions
            WHERE status = 'approved' AND executed_at IS NULL
            """
        ).fetchall()
        for candidate in candidates:
            _expire_locked(conn, int(candidate["id"]), candidate["expires_at"])
        conn.commit()
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


def set_pending_execution_outcome(
    action_id: int,
    *,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> bool:
    if status not in {"succeeded", "failed", "partial", "unknown", "expired"}:
        raise ValueError("invalid_execution_outcome")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE pending_actions
            SET execution_status = ?, execution_result_json = COALESCE(?, execution_result_json),
                execution_error = ?, executed_at = CASE
                    WHEN ? = 'succeeded' THEN COALESCE(executed_at, ?)
                    ELSE executed_at
                END
            WHERE id = ?
            """,
            (
                status,
                json.dumps(result, ensure_ascii=False, default=str) if result is not None else None,
                str(error)[:1000] if error else None,
                status,
                local_time_string(),
                action_id,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1

def delete_pending_action(action_id: int) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pending_actions WHERE id = ?", (action_id,))
        conn.commit()
        return cursor.rowcount > 0

