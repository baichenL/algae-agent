from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from app.core.db import assistant_conversations, conversation_tasks
from app.core.workspaces import current_workspace
from app.services.agent_runtime.state import RuntimeRequestContext


@dataclass(frozen=True)
class ResolvedConversationTask:
    task: dict[str, Any] | None
    request_context: RuntimeRequestContext
    reused_active: bool
    persisted: bool = True


def default_request_context(conversation_id: str) -> RuntimeRequestContext:
    conversation = assistant_conversations.get_conversation(conversation_id)
    workspace = current_workspace()
    return RuntimeRequestContext(
        owner=str((conversation or {}).get("owner") or "legacy"),
        role="scientist",
        workspace_id=str((conversation or {}).get("workspace_id") or workspace.id),
        conversation_id=conversation_id,
    )


def resolve_conversation_task(
    *,
    conversation_id: str,
    message: str,
    agent_run_id: str,
    request_context: RuntimeRequestContext | None = None,
) -> ResolvedConversationTask:
    context = request_context or default_request_context(conversation_id)
    if not assistant_conversations.get_conversation(conversation_id):
        return ResolvedConversationTask(
            task=None,
            request_context=context,
            reused_active=False,
            persisted=False,
        )
    active = conversation_tasks.get_active_task(
        conversation_id,
        owner=context.owner,
        workspace_id=context.workspace_id,
    )
    if active:
        task = conversation_tasks.update_task(
            active["id"],
            latest_agent_run_id=agent_run_id,
            event_type="task_turn_received",
        )
        return ResolvedConversationTask(
            task=task,
            request_context=replace(context, task_id=task["id"]),
            reused_active=True,
        )
    task = conversation_tasks.create_task(
        conversation_id=conversation_id,
        owner=context.owner,
        workspace_id=context.workspace_id,
        goal_text=message,
        latest_agent_run_id=agent_run_id,
    )
    return ResolvedConversationTask(
        task=task,
        request_context=replace(context, task_id=task["id"]),
        reused_active=False,
    )


def _pending_action(output: dict[str, Any]) -> dict[str, Any]:
    return {
        key: output.get(key)
        for key in ("action", "operation", "tool_name", "strain_id", "risk_level", "requires_approval")
        if output.get(key) is not None
    }


def synchronize_task_from_response(
    resolved: ResolvedConversationTask,
    response: Any,
) -> dict[str, Any]:
    if not resolved.persisted or not resolved.task:
        return {}
    task = conversation_tasks.get_task(resolved.task["id"]) or resolved.task
    output = dict(getattr(response, "agent_output", None) or {})
    action = str(output.get("action") or "")
    status = str(output.get("status") or "")
    missing = list(output.get("missing_fields") or [])

    if action in {"workflow_request_cancelled", "pending_form_cancelled"} or status == "cancelled":
        return conversation_tasks.cancel_task(task["id"])

    if status == "needs_more_info" or action in {"workflow_request_clarification", "require_more_info"}:
        task_type = "subculture" if action.startswith("workflow") else f"strain_{output.get('operation') or 'mutation'}"
        return conversation_tasks.update_task(
            task["id"],
            task_type=task_type,
            status="collecting",
            state_changing=True,
            missing_slots=missing,
            proposed_action=_pending_action(output),
            event_type="task_waiting_input",
        )

    pending_id = output.get("pending_id")
    if pending_id and status == "pending":
        task_type = "subculture" if action == "workflow_subculture" else f"strain_{output.get('operation') or 'mutation'}"
        return conversation_tasks.update_task(
            task["id"],
            task_type=task_type,
            status="waiting_approval",
            state_changing=True,
            missing_slots=[],
            pending_id=int(pending_id),
            proposed_action=_pending_action(output),
            event_type="task_pending_created",
        )

    if resolved.reused_active and task.get("state_changing"):
        conversation_tasks.append_task_event(
            task["id"],
            "task_side_turn_completed",
            {"action": action, "response_status": status},
        )
        return conversation_tasks.get_task(task["id"]) or task

    final_status = "failed" if status == "error" else "completed"
    return conversation_tasks.update_task(
        task["id"],
        status=final_status,
        event_type="task_failed" if final_status == "failed" else "task_completed",
    )


def task_for_pending(pending_id: int) -> dict[str, Any] | None:
    from app.core.db.connection import connect

    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT id FROM conversation_tasks WHERE pending_id = ? ORDER BY updated_at DESC LIMIT 1",
            (int(pending_id),),
        ).fetchone()
    return conversation_tasks.get_task(row["id"]) if row else None


def task_for_workflow_run(workflow_run_id: str) -> dict[str, Any] | None:
    from app.core.db.connection import connect

    canonical = workflow_run_id if workflow_run_id.startswith("workflow:") else f"workflow:{workflow_run_id}"
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT id FROM conversation_tasks WHERE workflow_run_id IN (?, ?) ORDER BY updated_at DESC LIMIT 1",
            (canonical, canonical.split(":", 1)[1]),
        ).fetchone()
    return conversation_tasks.get_task(row["id"]) if row else None


def update_task_for_pending(pending_id: int, *, status: str, workflow_run_id: str | None = None) -> dict[str, Any] | None:
    task = task_for_pending(pending_id)
    if not task:
        return None
    return conversation_tasks.update_task(
        task["id"],
        expected_version=int(task["version"]),
        status=status,
        workflow_run_id=workflow_run_id,
        event_type=f"task_{status}",
    )
