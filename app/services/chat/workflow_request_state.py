from __future__ import annotations

import datetime
import os
from copy import deepcopy
from typing import Any

from app.core.config import MEMORY_DIR
from app.core.db import assistant_conversations, conversation_tasks
from app.core.workspaces import current_workspace


STATE_DIR = os.path.join(MEMORY_DIR, "workflow_request_state")
DEFAULT_TTL_MINUTES = 24 * 60
_LEGACY_STATES: dict[str, dict[str, Any]] = {}


def _state_path(session_id: str) -> str:
    """Legacy path retained for one-way startup import only."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
    return os.path.join(STATE_DIR, f"{safe or 'default_session'}.json")


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def build_workflow_request_state(
    session_id: str,
    agent_run_id: str,
    source_message: str,
) -> dict[str, Any]:
    now = _now()
    return {
        "active": True,
        "state": "collecting_target",
        "workflow_name": "subculture",
        "required_slots": ["strain_id"],
        "collected_slots": {},
        "session_id": session_id,
        "agent_run_id": agent_run_id,
        "source_message": source_message,
        "created_at": now.isoformat(),
        "expires_at": (now + datetime.timedelta(minutes=DEFAULT_TTL_MINUTES)).isoformat(),
    }


def _task_for_state(session_id: str, state: dict[str, Any]) -> dict[str, Any]:
    task = None
    if state.get("agent_run_id"):
        task = conversation_tasks.get_task_for_run(str(state["agent_run_id"]))
    task = task or conversation_tasks.get_latest_task(session_id)
    if task:
        return task
    conversation = assistant_conversations.get_conversation(session_id) or {}
    return conversation_tasks.create_task(
        conversation_id=session_id,
        owner=str(conversation.get("owner") or "legacy"),
        workspace_id=str(conversation.get("workspace_id") or current_workspace().id),
        goal_text=str(state.get("source_message") or "传代 Workflow"),
        latest_agent_run_id=state.get("agent_run_id"),
    )


def save_workflow_request_state(session_id: str, state: dict[str, Any]) -> None:
    payload = deepcopy(state)
    if not assistant_conversations.get_conversation(session_id):
        _LEGACY_STATES[session_id] = payload
        return
    task = _task_for_state(session_id, payload)
    collected = dict(payload.get("collected_slots") or {})
    missing = list(payload.get("required_slots") or [])
    missing = [slot for slot in missing if not collected.get(slot)]
    status = "collecting" if missing else "ready"
    conversation_tasks.update_task(
        task["id"],
        task_type="subculture",
        goal_text=str(payload.get("source_message") or task.get("goal_text") or "传代 Workflow"),
        status=status,
        state_changing=True,
        collected_slots=collected,
        missing_slots=missing,
        proposed_action={
            "action": "trigger_subculture_workflow",
            "workflow_name": "subculture",
            "source_message": payload.get("source_message"),
            "request_agent_run_id": payload.get("agent_run_id"),
        },
        expires_at=payload.get("expires_at"),
        event_type="workflow_request_state_saved",
    )


def get_workflow_request_state(session_id: str) -> dict[str, Any] | None:
    if not assistant_conversations.get_conversation(session_id):
        return deepcopy(_LEGACY_STATES.get(session_id))
    task = conversation_tasks.get_active_task(session_id)
    if not task or task.get("task_type") != "subculture" or task.get("status") not in {"collecting", "ready"}:
        return None
    proposed = task.get("proposed_action") or {}
    return {
        "active": True,
        "state": "collecting_target" if task["status"] == "collecting" else "ready_to_create",
        "workflow_name": "subculture",
        "required_slots": list(task.get("missing_slots") or []),
        "collected_slots": dict(task.get("collected_slots") or {}),
        "session_id": session_id,
        "agent_run_id": proposed.get("request_agent_run_id") or task.get("latest_agent_run_id"),
        "source_message": proposed.get("source_message") or task.get("goal_text"),
        "created_at": task.get("created_at"),
        "expires_at": task.get("expires_at"),
        "task_id": task.get("id"),
        "task_version": task.get("version"),
    }


def clear_workflow_request_state(session_id: str) -> None:
    if not assistant_conversations.get_conversation(session_id):
        _LEGACY_STATES.pop(session_id, None)
        return
    task = conversation_tasks.get_active_task(session_id)
    if task and task.get("task_type") == "subculture":
        conversation_tasks.update_task(
            task["id"],
            expected_version=int(task["version"]),
            status="completed",
            event_type="workflow_request_state_consumed",
        )
