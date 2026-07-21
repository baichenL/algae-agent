import os
from copy import deepcopy
from typing import Any, Dict, Optional

from app.core.config import MEMORY_DIR
from app.core.db import assistant_conversations, conversation_tasks
from app.core.workspaces import current_workspace
from app.services.intent.routing_models import PendingFormDecision
from app.services.intent.write_action_parser import parse_pending_field_updates, validate_required_fields


STATE_DIR = os.path.join(MEMORY_DIR, "pending_form_state")
WRITE_OPERATION_LABELS = {"add": "新增品系", "update": "更新品系", "delete": "删除品系"}
_LEGACY_STATES: dict[str, dict[str, Any]] = {}


def get_state_file_path(session_id: str) -> str:
    """Legacy path retained for one-way startup import only."""
    safe_session_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
    return os.path.join(STATE_DIR, f"{safe_session_id}.json")


def get_pending_form_state(session_id: str) -> Optional[Dict[str, Any]]:
    if not assistant_conversations.get_conversation(session_id):
        return deepcopy(_LEGACY_STATES.get(session_id))
    task = conversation_tasks.get_active_task(session_id)
    if not task or not str(task.get("task_type") or "").startswith("strain_"):
        return None
    proposed = task.get("proposed_action") or {}
    return {
        "active": True,
        "operation": proposed.get("operation"),
        "tool_name": proposed.get("tool_name"),
        "collected_fields": dict(task.get("collected_slots") or {}),
        "missing_fields": list(task.get("missing_slots") or []),
        "candidates": list(proposed.get("candidates") or []),
        "source_message": proposed.get("source_message") or task.get("goal_text"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "task_id": task.get("id"),
        "task_version": task.get("version"),
    }


def save_pending_form_state(session_id: str, state: Dict[str, Any]) -> None:
    payload = deepcopy(state)
    if not assistant_conversations.get_conversation(session_id):
        _LEGACY_STATES[session_id] = payload
        return
    task = conversation_tasks.get_latest_task(session_id)
    if not task:
        conversation = assistant_conversations.get_conversation(session_id) or {}
        task = conversation_tasks.create_task(
            conversation_id=session_id,
            owner=str(conversation.get("owner") or "legacy"),
            workspace_id=str(conversation.get("workspace_id") or current_workspace().id),
            goal_text=str(payload.get("source_message") or "品系写操作"),
        )
    operation = str(payload.get("operation") or "mutation")
    missing = list(payload.get("missing_fields") or [])
    conversation_tasks.update_task(
        task["id"],
        task_type=f"strain_{operation}",
        goal_text=str(payload.get("source_message") or task.get("goal_text") or "品系写操作"),
        status="collecting" if missing else "ready",
        state_changing=True,
        collected_slots=dict(payload.get("collected_fields") or {}),
        missing_slots=missing,
        proposed_action={
            "operation": payload.get("operation"),
            "tool_name": payload.get("tool_name"),
            "candidates": list(payload.get("candidates") or []),
            "source_message": payload.get("source_message"),
        },
        event_type="pending_form_state_saved",
    )


def clear_pending_form_state(session_id: str) -> None:
    if not assistant_conversations.get_conversation(session_id):
        _LEGACY_STATES.pop(session_id, None)
        return
    task = conversation_tasks.get_active_task(session_id)
    if task and str(task.get("task_type") or "").startswith("strain_"):
        conversation_tasks.update_task(
            task["id"],
            expected_version=int(task["version"]),
            status="completed",
            event_type="pending_form_state_consumed",
        )


def build_pending_form_state(decision: PendingFormDecision, source_message: str) -> Dict[str, Any]:
    fields = {key: value for key, value in (decision.arguments or {}).items() if value is not None}
    _, missing_fields = validate_required_fields(decision.operation, fields)
    return {
        "active": True,
        "operation": decision.operation,
        "tool_name": decision.tool_name,
        "collected_fields": fields,
        "missing_fields": missing_fields,
        "candidates": list(decision.form_candidates),
        "source_message": source_message,
    }


def merge_pending_form_fields(state: Dict[str, Any], message: str, context_snapshot: Any) -> Dict[str, Any]:
    next_state = deepcopy(state)
    operation = next_state.get("operation")
    fields = dict(next_state.get("collected_fields") or {})
    updates, candidates = parse_pending_field_updates(
        operation, message, context_snapshot, next_state.get("missing_fields") or []
    )
    fields.update(updates)
    if candidates:
        next_state["candidates"] = candidates
    _, missing_fields = validate_required_fields(operation, fields)
    next_state["collected_fields"] = fields
    next_state["missing_fields"] = missing_fields
    return next_state


def operation_label(operation: Optional[str]) -> str:
    return WRITE_OPERATION_LABELS.get(operation or "", "写操作")
