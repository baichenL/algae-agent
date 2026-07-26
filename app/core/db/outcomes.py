from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from typing import Any

from app.core.db import connection
from app.core.time_utils import local_time_string
from app.core.effects import (
    CanonicalExecutionStatus,
    EffectClass,
    StateObservation,
    stable_hash,
)
from app.core.workspaces import current_workspace


TRUSTED_EXECUTOR_IDENTITIES = frozenset(
    {
        "trusted_commit_worker",
        "trusted_email_worker",
        "trusted_hardware_worker",
        "trusted_simulation_worker",
    }
)


def _receipt_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: receipt.get(key)
        for key in (
            "receipt_id",
            "execution_idempotency_key",
            "tool_call_id",
            "agent_run_id",
            "workspace_id",
            "pending_id",
            "approval_version",
            "proposal_hash",
            "effect_class",
            "executor_identity",
            "status",
            "state_changes",
            "business_fact_changed",
            "external_effect_performed",
            "provider_reference",
            "resource_versions",
            "result",
            "created_at",
        )
    }


def create_effect_receipt(
    *,
    execution_idempotency_key: str,
    workspace_id: str,
    effect_class: EffectClass | str,
    executor_identity: str,
    status: CanonicalExecutionStatus | str,
    result: dict[str, Any],
    pending_id: int | None = None,
    approval_version: int | None = None,
    proposal_hash: str | None = None,
    tool_call_id: str | None = None,
    agent_run_id: str | None = None,
    state_changes: list[dict[str, Any]] | None = None,
    business_fact_changed: bool = False,
    external_effect_performed: bool = False,
    provider_reference: str | None = None,
    resource_versions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if executor_identity not in TRUSTED_EXECUTOR_IDENTITIES:
        raise PermissionError("untrusted_receipt_executor")
    effect = EffectClass(effect_class)
    if effect not in {
        EffectClass.ARTIFACT_WRITE,
        EffectClass.DOMAIN_FACT_COMMIT,
        EffectClass.EXTERNAL_ACTUATION,
    }:
        raise ValueError("invalid_receipt_effect_class")
    canonical_status = CanonicalExecutionStatus(status)
    receipt = {
        "receipt_id": str(uuid.uuid4()),
        "execution_idempotency_key": execution_idempotency_key,
        "tool_call_id": tool_call_id,
        "agent_run_id": agent_run_id,
        "workspace_id": workspace_id,
        "pending_id": pending_id,
        "approval_version": approval_version,
        "proposal_hash": proposal_hash,
        "effect_class": effect.value,
        "executor_identity": executor_identity,
        "status": canonical_status.value,
        "state_changes": list(state_changes or []),
        "business_fact_changed": bool(business_fact_changed),
        "external_effect_performed": bool(external_effect_performed),
        "provider_reference": provider_reference,
        "resource_versions": dict(resource_versions or {}),
        "result": dict(result or {}),
        "created_at": local_time_string(),
    }
    receipt["payload_hash"] = stable_hash(_receipt_payload(receipt))
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "SELECT * FROM effect_receipts WHERE execution_idempotency_key = ?",
            (execution_idempotency_key,),
        ).fetchone()
        if existing:
            return _decode_receipt(dict(existing))
        conn.execute(
            """
            INSERT INTO effect_receipts (
                receipt_id, execution_idempotency_key, tool_call_id, agent_run_id,
                workspace_id, pending_id, approval_version, proposal_hash,
                effect_class, executor_identity, status, state_changes_json,
                business_fact_changed, external_effect_performed, provider_reference,
                resource_versions_json, result_json, payload_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt["receipt_id"],
                execution_idempotency_key,
                tool_call_id,
                agent_run_id,
                workspace_id,
                pending_id,
                approval_version,
                proposal_hash,
                effect.value,
                executor_identity,
                canonical_status.value,
                json.dumps(receipt["state_changes"], ensure_ascii=False, default=str),
                int(receipt["business_fact_changed"]),
                int(receipt["external_effect_performed"]),
                provider_reference,
                json.dumps(receipt["resource_versions"], ensure_ascii=False, default=str),
                json.dumps(receipt["result"], ensure_ascii=False, default=str),
                receipt["payload_hash"],
                receipt["created_at"],
            ),
        )
        conn.commit()
    return receipt


def _decode_receipt(row: dict[str, Any]) -> dict[str, Any]:
    for source, target, fallback in (
        ("state_changes_json", "state_changes", []),
        ("resource_versions_json", "resource_versions", {}),
        ("result_json", "result", {}),
    ):
        try:
            row[target] = json.loads(row.get(source) or "null") or fallback
        except Exception:
            row[target] = fallback
    row["business_fact_changed"] = bool(row.get("business_fact_changed"))
    row["external_effect_performed"] = bool(row.get("external_effect_performed"))
    return row


def get_effect_receipt(receipt_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM effect_receipts WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchone()
    return _decode_receipt(dict(row)) if row else None


def get_effect_receipt_by_execution_key(
    execution_idempotency_key: str,
) -> dict[str, Any] | None:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT * FROM effect_receipts
            WHERE execution_idempotency_key = ?
            """,
            (execution_idempotency_key,),
        ).fetchone()
    return _decode_receipt(dict(row)) if row else None


def get_state_observation(receipt_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM state_observations WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchone()
    return dict(row) if row else None


def reduce_effect_receipt(receipt_id: str) -> StateObservation:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        raw = conn.execute(
            "SELECT * FROM effect_receipts WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchone()
        if not raw:
            conn.rollback()
            raise KeyError("receipt_not_found")
        receipt = _decode_receipt(dict(raw))
        existing = conn.execute(
            "SELECT * FROM state_observations WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchone()
        if existing:
            conn.commit()
            return StateObservation(
                state_observation_id=existing["state_observation_id"],
                canonical_status=existing["canonical_status"],
                pending_status=existing["pending_status"],
                workflow_status=existing["workflow_status"],
                business_state_version=existing["business_state_version"],
                external_effect_status=existing["external_effect_status"],
                receipt_ref=receipt_id,
                updated_at=existing["created_at"],
            )
        if receipt["executor_identity"] not in TRUSTED_EXECUTOR_IDENTITIES:
            conn.rollback()
            raise PermissionError("untrusted_receipt_executor")
        if receipt.get("workspace_id") != current_workspace().id:
            conn.rollback()
            raise PermissionError("cross_workspace_receipt")
        if stable_hash(_receipt_payload(receipt)) != receipt.get("payload_hash"):
            conn.execute(
                """
                UPDATE effect_receipts
                SET reduction_status = 'rejected', reduction_error = ?
                WHERE receipt_id = ?
                """,
                ("receipt_hash_mismatch", receipt_id),
            )
            conn.commit()
            raise ValueError("receipt_hash_mismatch")
        pending = None
        pending_status = None
        workflow_status = None
        if receipt.get("pending_id") is not None:
            pending = conn.execute(
                "SELECT * FROM pending_actions WHERE id = ?",
                (int(receipt["pending_id"]),),
            ).fetchone()
            if not pending:
                conn.rollback()
                raise ValueError("receipt_pending_not_found")
            if pending["status"] != "approved":
                conn.rollback()
                raise ValueError("receipt_pending_not_approved")
            if int(pending["approval_version"] or 0) != int(receipt.get("approval_version") or 0):
                conn.rollback()
                raise ValueError("receipt_approval_version_mismatch")
            expected_hash = pending["proposal_hash"]
            if expected_hash and expected_hash != receipt.get("proposal_hash"):
                conn.rollback()
                raise ValueError("receipt_proposal_hash_mismatch")
            result_json = json.dumps(receipt.get("result") or {}, ensure_ascii=False, default=str)
            execution_status = receipt["status"]
            executed_at = local_time_string() if execution_status == "succeeded" else None
            conn.execute(
                """
                UPDATE pending_actions
                SET execution_status = ?, execution_result_json = ?,
                    execution_error = CASE WHEN ? = 'failed' THEN ? ELSE NULL END,
                    executed_at = CASE
                        WHEN ? = 'succeeded' THEN COALESCE(executed_at, ?)
                        ELSE executed_at
                    END,
                    resume_completed_at = CASE
                        WHEN ? = 'succeeded' THEN COALESCE(resume_completed_at, ?)
                        ELSE resume_completed_at
                    END
                WHERE id = ?
                """,
                (
                    execution_status,
                    result_json,
                    execution_status,
                    str((receipt.get("result") or {}).get("reason") or "")[:1000],
                    execution_status,
                    executed_at,
                    execution_status,
                    executed_at,
                    int(receipt["pending_id"]),
                ),
            )
            pending_status = pending["status"]
            workflow = conn.execute(
                "SELECT status FROM workflow_runs WHERE pending_id = ?",
                (int(receipt["pending_id"]),),
            ).fetchone()
            workflow_status = workflow["status"] if workflow else None
        canonical_status = receipt["status"]
        external_status = (
            canonical_status
            if receipt["effect_class"] == EffectClass.EXTERNAL_ACTUATION.value
            else None
        )
        business_version = stable_hash(receipt.get("resource_versions") or {})[:16] or None
        observation = StateObservation(
            state_observation_id=str(uuid.uuid4()),
            canonical_status=canonical_status,
            pending_status=pending_status,
            workflow_status=workflow_status,
            business_state_version=business_version,
            external_effect_status=external_status,
            receipt_ref=receipt_id,
            updated_at=local_time_string(),
        )
        conn.execute(
            """
            INSERT INTO state_observations (
                state_observation_id, receipt_id, pending_id, canonical_status,
                pending_status, workflow_status, business_state_version,
                external_effect_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation.state_observation_id,
                receipt_id,
                receipt.get("pending_id"),
                observation.canonical_status,
                observation.pending_status,
                observation.workflow_status,
                observation.business_state_version,
                observation.external_effect_status,
                observation.updated_at,
            ),
        )
        conn.execute(
            """
            UPDATE effect_receipts
            SET reduced_at = ?, reduction_status = 'applied', reduction_error = NULL
            WHERE receipt_id = ?
            """,
            (observation.updated_at, receipt_id),
        )
        conn.commit()
        return observation
