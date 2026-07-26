from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from app.core.db import assistant_conversations, conversation_tasks
from app.core.workspaces import current_workspace
from app.services.agent_runtime.state import RuntimeRequestContext
from app.services.chat.task_spec import TaskSpec


@dataclass(frozen=True)
class ResolvedConversationTask:
    task: dict[str, Any] | None
    request_context: RuntimeRequestContext
    reused_active: bool
    persisted: bool = True
    task_spec: TaskSpec | None = None


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
    task_spec: TaskSpec | None = None,
) -> ResolvedConversationTask:
    context = request_context or default_request_context(conversation_id)
    if not assistant_conversations.get_conversation(conversation_id):
        return ResolvedConversationTask(
            task=None,
            request_context=context,
            reused_active=False,
            persisted=False,
            task_spec=task_spec,
        )
    active = conversation_tasks.get_active_task(
        conversation_id,
        owner=context.owner,
        workspace_id=context.workspace_id,
    )
    relation = task_spec.task_relation if task_spec else "continue_current"
    domain = task_spec.task_domain if task_spec else "conversation_turn"
    if active and relation == "continue_current":
        task = conversation_tasks.update_task(
            active["id"],
            latest_agent_run_id=agent_run_id,
            task_spec=task_spec.to_dict() if task_spec else active.get("task_spec"),
            event_type="task_turn_received",
        )
        return ResolvedConversationTask(
            task=task,
            request_context=replace(context, task_id=task["id"]),
            reused_active=True,
            task_spec=task_spec,
        )
    if active and relation == "cancel_previous_and_start":
        conversation_tasks.cancel_task(active["id"], reason="superseded_by_explicit_user_request")
        active = None
    elif active and relation == "start_new":
        conversation_tasks.suspend_task(active["id"])
        active = None
    elif relation == "resume_named_task":
        named = conversation_tasks.find_latest_task(
            conversation_id,
            task_type=domain,
            owner=context.owner,
            workspace_id=context.workspace_id,
        )
        if named:
            if active and active["id"] != named["id"]:
                conversation_tasks.suspend_task(active["id"], superseded_by_task_id=named["id"])
            task = conversation_tasks.update_task(
                named["id"],
                status="running",
                state_changing=domain in {"email", "strain_mutation", "subculture"},
                latest_agent_run_id=agent_run_id,
                task_spec=task_spec.to_dict() if task_spec else named.get("task_spec"),
                event_type="task_resumed",
            )
            return ResolvedConversationTask(
                task=task,
                request_context=replace(context, task_id=task["id"]),
                reused_active=True,
                task_spec=task_spec,
            )
    task = conversation_tasks.create_task(
        conversation_id=conversation_id,
        owner=context.owner,
        workspace_id=context.workspace_id,
        goal_text=message,
        task_type=domain,
        latest_agent_run_id=agent_run_id,
        task_spec=task_spec.to_dict() if task_spec else None,
        parent_task_id=active["id"] if active and relation == "side_question" else None,
    )
    return ResolvedConversationTask(
        task=task,
        request_context=replace(context, task_id=task["id"]),
        reused_active=False,
        task_spec=task_spec,
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
        is_email = (
            (resolved.task_spec and resolved.task_spec.task_domain == "email")
            or action == "email_draft"
        )
        task_type = (
            "email"
            if is_email
            else "subculture"
            if action.startswith("workflow")
            else f"strain_{output.get('operation') or 'mutation'}"
        )
        return conversation_tasks.update_task(
            task["id"],
            task_type=task_type,
            status="collecting",
            state_changing=True,
            missing_slots=missing,
            proposed_action={
                **_pending_action(output),
                **(
                    {"email_request_spec": output.get("email_request_spec")}
                    if output.get("email_request_spec")
                    else {}
                ),
            },
            event_type="task_waiting_input",
        )

    pending_id = output.get("pending_id")
    if pending_id and status == "pending":
        if action == "email_draft":
            task_type = "email"
        else:
            task_type = "subculture" if action == "workflow_subculture" else f"strain_{output.get('operation') or 'mutation'}"
        return conversation_tasks.update_task(
            task["id"],
            task_type=task_type,
            status="waiting_approval",
            state_changing=True,
            missing_slots=[],
            pending_id=int(pending_id),
            proposed_action={
                **_pending_action(output),
                **({"draft": output.get("draft")} if output.get("draft") else {}),
            },
            event_type="task_pending_created",
        )

    if action == "email_draft" and status == "success":
        return conversation_tasks.update_task(
            task["id"],
            task_type="email",
            status="ready",
            state_changing=True,
            missing_slots=[],
            proposed_action={
                **_pending_action(output),
                "draft": output.get("draft"),
                "requires_approval": True,
            },
            event_type="task_email_draft_ready",
        )

    outcome_status = str(
        output.get("outcome_status")
        or (output.get("answer_envelope") or {}).get("outcome_status")
        or ""
    )
    if status == "paused" or outcome_status == "paused" or (
        status == "partial" and output.get("budget_exhausted")
    ):
        return conversation_tasks.pause_task(
            task["id"],
            reason=str(output.get("pause_reason") or output.get("budget_exhausted") or "budget_exhausted"),
            proposed_action={
                **(task.get("proposed_action") or {}),
                "checkpoint_id": output.get("checkpoint_id"),
                "remaining_work": output.get("remaining_work") or [],
                "budget": output.get("budget") or {},
            },
        )

    if resolved.reused_active and task.get("state_changing"):
        conversation_tasks.append_task_event(
            task["id"],
            "task_side_turn_completed",
            {"action": action, "response_status": status},
        )
        return conversation_tasks.get_task(task["id"]) or task

    final_status = "failed" if status in {"error", "failed"} or outcome_status == "failed" else "completed"
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
