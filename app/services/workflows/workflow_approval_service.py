from __future__ import annotations

import os

from app.core import database
from app.core.db import workflow_runs
from app.services.agent_runtime import record_run_event
from app.services.memory.memory_service import append_decision_event
from app.services.observability.error_events import record_error_event
from app.services.protocols.pending_payload import verify_frozen_protocol_payload
from app.services.workflows.workflow_execution_service import execute_workflow_run


def grant_workflow_approval(
    pending_id: int,
    *,
    reviewed_by: str = "human",
) -> dict:
    pending = database.get_pending_action(pending_id)
    if not pending:
        record_error_event(
            layer="pending_approval",
            component="workflow_approval_service",
            operation="grant_workflow_approval",
            severity="warning",
            error_type="PendingNotFound",
            error_message="Workflow pending action was not found.",
            metadata={"pending_id": pending_id},
        )
        return {"status": "error", "reason": "not_found"}
    payload = pending.get("payload") or {}
    data = payload.get("data") or {}
    if payload.get("type") != "workflow_subculture":
        record_error_event(
            session_id=data.get("session_id"),
            agent_run_id=data.get("agent_run_id"),
            layer="pending_approval",
            component="workflow_approval_service",
            operation="grant_workflow_approval",
            severity="warning",
            error_type="NotWorkflowPending",
            error_message="Pending action is not a workflow_subculture request.",
            metadata={"pending_id": pending_id, "action_type": payload.get("type")},
        )
        return {"status": "error", "reason": "not_workflow_pending"}

    protocol_validation = verify_frozen_protocol_payload(payload)
    if not protocol_validation.get("valid"):
        append_decision_event({
            "event": "protocol_hash_mismatch",
            "agent_run_id": data.get("agent_run_id"),
            "session_id": data.get("session_id"),
            "pending_id": pending_id,
            "strain_id": data.get("strain_id"),
            "reason": protocol_validation.get("reason"),
            "protocol_id": protocol_validation.get("protocol_id"),
            "protocol_hash": protocol_validation.get("protocol_hash"),
        })
        record_error_event(
            session_id=data.get("session_id"),
            agent_run_id=data.get("agent_run_id"),
            layer="pending_approval",
            component="workflow_approval_service",
            operation="verify_frozen_protocol_payload",
            severity="error",
            error_type="ProtocolHashMismatch",
            error_message="Frozen protocol payload failed hash verification.",
            metadata={"pending_id": pending_id, **protocol_validation},
        )
        return {"status": "error", "reason": "protocol_hash_mismatch"}

    existing_run = workflow_runs.get_workflow_run_by_pending(pending_id)
    if pending.get("status") == "approved" and existing_run:
        return {
            "status": "success",
            "action": "approved",
            "executed": existing_run.get("status") == "succeeded",
            "pending_id": pending_id,
            "run_id": existing_run.get("id"),
            "workflow_run": existing_run,
            "idempotent": True,
        }
    if pending.get("status") != "pending":
        record_error_event(
            session_id=data.get("session_id"),
            agent_run_id=data.get("agent_run_id"),
            layer="pending_approval",
            component="workflow_approval_service",
            operation="grant_workflow_approval",
            severity="warning",
            error_type="PendingAlreadyReviewed",
            error_message="Workflow pending action has already been reviewed.",
            metadata={"pending_id": pending_id, "status": pending.get("status")},
        )
        return {"status": "error", "reason": "already_reviewed"}

    strain_id = data.get("strain_id")
    current = database.get_algae_status(strain_id)
    if not current:
        append_decision_event({
            "event": "pending_review_failed",
            "agent_run_id": data.get("agent_run_id"),
            "session_id": data.get("session_id"),
            "pending_id": pending_id,
            "action_type": "workflow_subculture",
            "strain_id": strain_id,
            "reason": "strain_not_found",
            "executed": False,
        })
        record_error_event(
            session_id=data.get("session_id"),
            agent_run_id=data.get("agent_run_id"),
            layer="pending_approval",
            component="workflow_approval_service",
            operation="grant_workflow_approval",
            severity="error",
            error_type="StrainNotFound",
            error_message="Cannot approve workflow pending request because strain was not found.",
            metadata={"pending_id": pending_id, "strain_id": strain_id},
        )
        return {"status": "error", "reason": "strain_not_found"}
    frozen_protocol = payload.get("protocol") or {}
    expected_generation = frozen_protocol.get("expected_generation", data.get("expected_generation"))
    if (
        expected_generation is not None
        and int(current.get("generation_number") or 0) != int(expected_generation)
    ):
        append_decision_event({
            "event": "pending_review_failed",
            "pending_id": pending_id,
            "action_type": "workflow_subculture",
            "strain_id": strain_id,
            "reason": "stale_generation",
            "executed": False,
        })
        record_error_event(
            session_id=data.get("session_id"),
            agent_run_id=data.get("agent_run_id"),
            layer="pending_approval",
            component="workflow_approval_service",
            operation="grant_workflow_approval",
            severity="error",
            error_type="StaleGeneration",
            error_message="Cannot approve workflow pending request because generation is stale.",
            metadata={
                "pending_id": pending_id,
                "strain_id": strain_id,
                "expected_generation": expected_generation,
                "current_generation": current.get("generation_number"),
            },
        )
        return {"status": "error", "reason": "stale_generation"}
    mode = os.getenv("HARDWARE_MODE", "simulation").strip().lower()
    try:
        run, idempotent = workflow_runs.approve_pending_and_create_run(
            pending_id=pending_id,
            reviewed_by=reviewed_by,
            workflow_name=data.get("workflow") or "subculture",
            strain_id=strain_id,
            execution_mode=mode,
            expected_generation=(
                int(expected_generation)
                if expected_generation is not None
                else int(current.get("generation_number") or 0)
            ),
        )
    except ValueError as exc:
        record_error_event(
            session_id=data.get("session_id"),
            agent_run_id=data.get("agent_run_id"),
            layer="pending_approval",
            component="workflow_approval_service",
            operation="approve_pending_and_create_run",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"pending_id": pending_id, "strain_id": strain_id},
        )
        return {"status": "error", "reason": str(exc)}
    append_decision_event({
        "event": "workflow_approval_granted",
        "agent_run_id": data.get("agent_run_id"),
        "session_id": data.get("session_id"),
        "pending_id": pending_id,
        "workflow_run_id": run.get("id"),
        "strain_id": strain_id,
        "reviewed_by": reviewed_by,
    })
    append_decision_event({
        "event": "approval_granted",
        "agent_run_id": data.get("agent_run_id"),
        "session_id": data.get("session_id"),
        "pending_id": pending_id,
        "workflow_run_id": run.get("id"),
        "strain_id": strain_id,
        "protocol_id": protocol_validation.get("protocol_id"),
        "protocol_hash": protocol_validation.get("protocol_hash"),
        "reviewed_by": reviewed_by,
    })
    record_run_event(
        data.get("agent_run_id"),
        session_id=data.get("session_id"),
        event_type="workflow_queued",
        layer="langgraph_workflow",
        payload={
            "pending_id": pending_id,
            "workflow_run_id": run.get("id"),
            "strain_id": strain_id,
            "idempotent": idempotent,
        },
    )
    return {
        "status": "success",
        "action": "approved",
        "executed": False,
        "pending_id": pending_id,
        "run_id": run.get("id"),
        "action_type": "workflow_subculture",
        "strain_id": strain_id,
        "run_status": run.get("status"),
        "idempotent": idempotent,
        "msg": f"审批已通过，Workflow Run {run.get('id')} 已进入执行队列。",
    }


async def approve_workflow_pending(
    pending_id: int,
    *,
    reviewed_by: str = "human",
) -> dict:
    approval = grant_workflow_approval(pending_id, reviewed_by=reviewed_by)
    if approval.get("status") != "success":
        return approval
    if approval.get("idempotent"):
        return approval

    execution = await execute_workflow_run(approval["run_id"])
    if execution.get("status") != "success":
        return {
            "status": "error",
            "reason": execution.get("reason") or "workflow_failed",
            "pending_id": pending_id,
            "run_id": approval.get("run_id"),
            "workflow_result": execution,
        }
    simulation = not execution.get("physical_execution")
    return {
        "status": "success",
        "action": "approved",
        "executed": True,
        "pending_id": pending_id,
        "run_id": approval.get("run_id"),
        "action_type": "workflow_subculture",
        "strain_id": approval.get("strain_id"),
        "workflow_result": execution,
        "physical_execution": execution.get("physical_execution", False),
        "persisted": execution.get("persisted", False),
        "msg": (
            f"已审批并完成传代模拟流程: {approval.get('strain_id')}；正式藻种状态未更新。"
            if simulation
            else f"已审批并完成物理传代流程: {approval.get('strain_id')}。"
        ),
    }
