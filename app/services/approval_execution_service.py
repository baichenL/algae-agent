from __future__ import annotations

from typing import Any

from app.core import database
from app.core.db import operations, scientific as scientific_db, workflow_runs
from app.core.workspaces import WorkspaceContext, current_workspace, workspace_scope
from app.services.agent_runtime.contracts_v2 import (
    CanonicalExecutionStatus,
    EffectClass,
)
from app.services.chat.task_coordinator import update_task_for_pending
from app.services.scientific.service import approve_and_simulate_proposal
from app.services.proposal_execution_validation import validate_pending_for_execution
from app.models.email_schema import EmailSendRequest
from app.services.email.email_service import send_email
from app.services.strains import strain_service
from app.services.workflows.workflow_execution_service import execute_workflow_run
from app.services.agent_runtime.events import record_run_event


def _execution_key(pending: dict, approval_id: int) -> str:
    return str(
        pending.get("execution_idempotency_key")
        or f"pending:{approval_id}:approval:{int(pending.get('approval_version') or 0)}"
    )


def _effect_for_action(action_type: str) -> EffectClass:
    if action_type in {"add_strain", "update_strain", "delete_strain"}:
        return EffectClass.DOMAIN_FACT_COMMIT
    if action_type == "email_send":
        return EffectClass.EXTERNAL_ACTUATION
    return EffectClass.ARTIFACT_WRITE


def _state_observation_payload(observation) -> dict:
    return {
        "state_observation_id": observation.state_observation_id,
        "canonical_status": observation.canonical_status,
        "pending_status": observation.pending_status,
        "workflow_status": observation.workflow_status,
        "business_state_version": observation.business_state_version,
        "external_effect_status": observation.external_effect_status,
        "receipt_ref": observation.receipt_ref,
        "updated_at": observation.updated_at,
    }


def _record_outcome(
    *,
    pending: dict,
    approval_id: int,
    action_type: str,
    data: dict,
    result: dict,
) -> dict:
    result_status = str(result.get("status") or "failed")
    canonical_status = {
        "success": CanonicalExecutionStatus.SUCCEEDED,
        "succeeded": CanonicalExecutionStatus.SUCCEEDED,
        "partial": CanonicalExecutionStatus.PARTIAL,
        "unknown": CanonicalExecutionStatus.UNKNOWN,
    }.get(result_status, CanonicalExecutionStatus.FAILED)
    succeeded = canonical_status == CanonicalExecutionStatus.SUCCEEDED
    effect_class = _effect_for_action(action_type)
    receipt = database.create_effect_receipt(
        execution_idempotency_key=_execution_key(pending, approval_id),
        workspace_id=current_workspace().id,
        effect_class=effect_class,
        executor_identity=(
            "trusted_commit_worker"
            if effect_class == EffectClass.DOMAIN_FACT_COMMIT
            else (
                "trusted_email_worker"
                if effect_class == EffectClass.EXTERNAL_ACTUATION
                else "trusted_simulation_worker"
            )
        ),
        status=canonical_status,
        result=result,
        pending_id=approval_id,
        approval_version=int(pending.get("approval_version") or 0),
        proposal_hash=pending.get("proposal_hash"),
        agent_run_id=pending.get("agent_run_id"),
        state_changes=(
            [{
                "entity": "strain",
                "entity_id": data.get("strain_id"),
                "operation": action_type,
            }]
            if succeeded and effect_class == EffectClass.DOMAIN_FACT_COMMIT
            else []
        ),
        business_fact_changed=bool(
            succeeded and effect_class == EffectClass.DOMAIN_FACT_COMMIT
        ),
        external_effect_performed=bool(
            effect_class == EffectClass.EXTERNAL_ACTUATION and result.get("sent")
        ),
        provider_reference=result.get("provider_reference"),
        resource_versions={
            "strain_id": data.get("strain_id"),
            "expected_generation": data.get("expected_generation"),
        },
    )
    record_run_event(
        pending.get("agent_run_id"),
        event_type="effect_receipt_created",
        layer="outcome_reducer",
        payload={
            "receipt_id": receipt["receipt_id"],
            "pending_id": approval_id,
            "status": receipt["status"],
            "effect_class": receipt["effect_class"],
        },
    )
    observation = database.reduce_effect_receipt(receipt["receipt_id"])
    record_run_event(
        pending.get("agent_run_id"),
        event_type="canonical_state_updated",
        layer="outcome_reducer",
        payload=_state_observation_payload(observation),
    )
    return {
        **result,
        "receipt_id": receipt["receipt_id"],
        "state_observation": _state_observation_payload(observation),
    }


async def execute_approved_pending(
    approval_id: int,
    workflow_run_id: int | None = None,
    workspace: WorkspaceContext | None = None,
    operation_id: str | None = None,
) -> dict:
    """Trusted execution entrypoint.

    API and Agent Runtime may queue an approved pending action, but only this
    worker entrypoint interprets it as an execution request. The returned
    result is not canonical until the receipt has been reduced.
    """
    if workspace and workspace.id != current_workspace().id:
        with workspace_scope(workspace):
            return await execute_approved_pending(
                approval_id,
                workflow_run_id,
                operation_id=operation_id,
            )
    if operation_id:
        operations.update_operation(
            operation_id,
            status="running",
            phase="starting",
            message="Approved proposal claimed by trusted worker",
            progress=0.08,
        )
    pending = database.get_pending_action(approval_id)
    if not pending:
        return {"status": "error", "reason": "approval_not_found"}
    if pending.get("status") == "expired":
        return {"status": "error", "reason": "approval_expired"}
    execution_key = _execution_key(pending, approval_id)
    existing_receipt = database.get_effect_receipt_by_execution_key(execution_key)
    if existing_receipt:
        receipt_id = existing_receipt["receipt_id"]
        observation = database.get_state_observation(receipt_id)
        return {
            "status": "success",
            "idempotent": True,
            "receipt_id": receipt_id,
            "state_observation": observation,
        }
    if pending.get("executed_at") and pending.get("execution_result"):
        return {
            "status": "success",
            "idempotent": True,
            "result": pending.get("execution_result"),
        }
    if pending.get("status") != "approved":
        return {"status": "error", "reason": "approval_not_approved"}
    stale_reason = validate_pending_for_execution(pending)
    if stale_reason:
        database.mark_pending_stale(approval_id, stale_reason)
        return {"status": "error", "reason": stale_reason, "pending_status": "stale"}
    if not database.claim_pending_execution(approval_id):
        current = database.get_pending_action(approval_id)
        if current and current.get("executed_at"):
            return {
                "status": "success",
                "idempotent": True,
                "result": current.get("execution_result"),
            }
        return {"status": "error", "reason": "execution_not_claimed"}
    record_run_event(
        pending.get("agent_run_id"),
        event_type="worker_claimed",
        layer="trusted_worker",
        payload={
            "pending_id": approval_id,
            "execution_idempotency_key": execution_key,
        },
    )
    update_task_for_pending(approval_id, status="running")

    payload = pending.get("payload") or {}
    action_type = payload.get("type") or pending.get("action_type")
    data = payload.get("data") or {}
    target_id = workflow_run_id
    try:
        if action_type == "workflow_subculture":
            run = workflow_runs.get_workflow_run_by_pending(approval_id)
            target_id = workflow_run_id or (int(run["id"]) if run else None)
            if target_id is None:
                raise RuntimeError("workflow_run_not_found")
            if run and run.get("status") == "succeeded":
                result = {"status": "success", "run_id": target_id, "idempotent": True}
            else:
                if run and run.get("status") == "running":
                    workflow_runs.requeue_interrupted_workflow_run(target_id)
                if operation_id:
                    operations.update_operation(
                        operation_id,
                        phase="simulation",
                        message="Running approved simulation",
                        progress=0.15,
                        related_run_id=f"workflow:{target_id}",
                    )
                result = await execute_workflow_run(target_id)
        elif action_type == "scientific_experiment_plan":
            result = approve_and_simulate_proposal(
                approval_id,
                reviewed_by=str(pending.get("reviewed_by") or "system"),
                idempotency_key=f"approval-auto-{approval_id}",
                finalize_pending=False,
            )
        elif action_type == "email_send":
            draft = data.get("draft") or {}
            message_id = (
                f"<algae-{stable_hash(execution_key)[:32]}@"
                f"{current_workspace().id}.local>"
            )
            email_response = send_email(
                EmailSendRequest(
                    subject=str(draft.get("subject") or ""),
                    body=str(draft.get("body") or ""),
                    recipients=list(draft.get("recipients") or []),
                    source="trusted_email_worker",
                    metadata={
                        **dict(draft.get("metadata") or {}),
                        "template_type": draft.get("template_type") or "manual_check",
                        "message_id": message_id,
                        "pending_id": approval_id,
                    },
                )
            )
            result = email_response.model_dump()
            result["provider_reference"] = message_id
            if (
                result.get("status") == "error"
                and "not configured" not in str(result.get("message") or "").lower()
            ):
                # SMTP can accept the message and lose the response. A fixed
                # Message-ID makes manual reconciliation possible, but the
                # worker must never auto-resend an ambiguous outcome.
                result["status"] = "unknown"
                result["reason"] = "email_delivery_outcome_ambiguous"
        elif action_type in {"add_strain", "update_strain", "delete_strain"}:
            result = await strain_service.execute_approved_pending_action(
                approval_id,
                execution_idempotency_key=execution_key,
                finalize_pending=False,
            )
        else:
            result = {"status": "error", "reason": "unsupported_action"}
    except Exception as exc:
        result = {"status": "error", "reason": str(exc)}

    projected = _record_outcome(
        pending=pending,
        approval_id=approval_id,
        action_type=action_type,
        data=data,
        result=result,
    )
    if result.get("status") not in {"success", "succeeded"}:
        update_task_for_pending(approval_id, status="failed")
        if action_type == "scientific_experiment_plan" and data.get("scientific_run_id"):
            scientific_db.update_scientific_run(
                str(data["scientific_run_id"]),
                status="simulation_failed",
            )
        if operation_id:
            ambiguous = result.get("status") == "unknown"
            operations.update_operation(
                operation_id,
                status="failed",
                phase="unknown" if ambiguous else "failed",
                message=(
                    "Execution outcome is unknown and requires manual reconciliation"
                    if ambiguous
                    else "Trusted execution failed"
                ),
                error_message=str(result.get("reason") or result),
                retryable=False if ambiguous else True,
            )
        return projected

    if action_type == "workflow_subculture":
        update_task_for_pending(
            approval_id,
            status="waiting_manual_confirmation",
            workflow_run_id=f"workflow:{target_id}",
        )
    else:
        update_task_for_pending(approval_id, status="completed")
    if operation_id:
        operations.update_operation(
            operation_id,
            status="succeeded",
            phase="completed",
            message="Trusted execution completed",
            progress=1.0,
            retryable=False,
        )
    return projected


async def recover_approved_executions() -> list[dict]:
    results = []
    for pending in database.list_recoverable_pending_executions():
        approval_id = int(pending["id"])
        if pending.get("execution_status") == "running":
            database.mark_pending_execution_failed(
                approval_id,
                "recovered_after_restart",
            )
        database.queue_pending_execution(approval_id)
        operation = operations.find_latest_related("approval", str(approval_id))
        operation_id = operation.get("id") if operation else None
        if operation_id:
            operations.update_operation(
                operation_id,
                status="queued",
                phase="recovered",
                message="Trusted worker recovered approved execution",
                progress=0.0,
                retryable=True,
            )
        results.append(
            await execute_approved_pending(
                approval_id,
                operation_id=operation_id,
            )
        )
    return results
