from __future__ import annotations

from app.core import database
from app.core.db import scientific as scientific_db
from app.core.db import workflow_runs, operations
from app.core.workspaces import WorkspaceContext, current_workspace, workspace_scope
from app.services.scientific.service import approve_and_simulate_proposal
from app.services.strains import strain_service
from app.services.workflows.workflow_execution_service import execute_workflow_run
from app.services.chat.task_coordinator import update_task_for_pending


async def execute_approved_pending(
    approval_id: int,
    workflow_run_id: int | None = None,
    workspace: WorkspaceContext | None = None,
    operation_id: str | None = None,
) -> dict:
    if workspace and workspace.id != current_workspace().id:
        with workspace_scope(workspace):
            return await execute_approved_pending(approval_id, workflow_run_id, operation_id=operation_id)
    if operation_id:
        operations.update_operation(
            operation_id,
            status="running",
            phase="starting",
            message="审批已通过，正在启动执行",
            progress=0.08,
        )
    pending = database.get_pending_action(approval_id)
    if not pending:
        return {"status": "error", "reason": "approval_not_found"}
    if pending.get("executed_at"):
        return {"status": "success", "idempotent": True, "result": pending.get("execution_result")}
    if pending.get("status") != "approved":
        return {"status": "error", "reason": "approval_not_approved"}
    if not database.claim_pending_execution(approval_id):
        current = database.get_pending_action(approval_id)
        if current and current.get("executed_at"):
            return {"status": "success", "idempotent": True, "result": current.get("execution_result")}
        return {"status": "error", "reason": "execution_not_claimed"}
    update_task_for_pending(approval_id, status="running")

    payload = pending.get("payload") or {}
    action_type = payload.get("type") or pending.get("action_type")
    data = payload.get("data") or {}
    try:
        if action_type == "workflow_subculture":
            run = workflow_runs.get_workflow_run_by_pending(approval_id)
            target_id = workflow_run_id or (int(run["id"]) if run else None)
            if target_id is None:
                raise RuntimeError("workflow_run_not_found")
            if run and run.get("status") == "succeeded":
                result = {"status": "success", "run_id": target_id, "idempotent": True}
                database.mark_pending_executed(approval_id, result)
                return result
            if run and run.get("status") == "running":
                workflow_runs.requeue_interrupted_workflow_run(target_id)
            if operation_id:
                operations.update_operation(
                    operation_id,
                    phase="simulation",
                    message="正在运行传代仿真",
                    progress=0.15,
                    related_run_id=f"workflow:{target_id}",
                )
            result = await execute_workflow_run(target_id)
        elif action_type == "scientific_experiment_plan":
            result = approve_and_simulate_proposal(
                approval_id,
                reviewed_by=str(pending.get("reviewed_by") or "system"),
                idempotency_key=f"approval-auto-{approval_id}",
            )
        elif action_type in {"add_strain", "update_strain", "delete_strain"}:
            result = await strain_service.execute_approved_pending_action(approval_id)
        else:
            result = {"status": "error", "reason": "unsupported_action"}
    except Exception as exc:
        result = {"status": "error", "reason": str(exc)}

    if result.get("status") != "success":
        update_task_for_pending(approval_id, status="failed")
        database.mark_pending_execution_failed(approval_id, str(result.get("reason") or result))
        if action_type == "scientific_experiment_plan" and data.get("scientific_run_id"):
            scientific_db.update_scientific_run(str(data["scientific_run_id"]), status="simulation_failed")
        if operation_id:
            operations.update_operation(
                operation_id,
                status="failed",
                phase="failed",
                message="自动执行失败",
                error_message=str(result.get("reason") or result),
                retryable=True,
            )
        return result
    database.mark_pending_executed(approval_id, result)
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
            message="自动执行已完成",
            progress=1.0,
            retryable=False,
        )
    return result


async def recover_approved_executions() -> list[dict]:
    results = []
    for pending in database.list_recoverable_pending_executions():
        approval_id = int(pending["id"])
        if pending.get("execution_status") == "running":
            database.mark_pending_execution_failed(approval_id, "recovered_after_restart")
        database.queue_pending_execution(approval_id)
        operation = operations.find_latest_related("approval", str(approval_id))
        operation_id = operation.get("id") if operation else None
        if operation_id:
            operations.update_operation(
                operation_id,
                status="queued",
                phase="recovered",
                message="服务已恢复，正在重新启动审批执行",
                progress=0.0,
                retryable=True,
            )
        results.append(await execute_approved_pending(approval_id, operation_id=operation_id))
    return results
