from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.db import assistant_conversations, conversation_tasks
from app.services.chat.pending_form_state import STATE_DIR as PENDING_FORM_DIR
from app.services.chat.workflow_request_state import STATE_DIR as WORKFLOW_REQUEST_DIR


logger = logging.getLogger(__name__)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) and payload.get("active") else None
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read legacy task state: %s", path)
        return None


def _conversation_id(path: Path, payload: dict[str, Any]) -> str:
    return str(payload.get("session_id") or path.stem)


def _import(path: Path, payload: dict[str, Any], *, task_type: str) -> bool:
    legacy_source = str(path.resolve())
    if conversation_tasks.get_task_by_legacy_source(legacy_source):
        return False
    conversation_id = _conversation_id(path, payload)
    conversation = assistant_conversations.get_conversation(conversation_id)
    if not conversation:
        logger.warning("Legacy task state has no matching conversation and was preserved: %s", path)
        return False
    if task_type == "subculture":
        collected = dict(payload.get("collected_slots") or {})
        missing = [slot for slot in (payload.get("required_slots") or ["strain_id"]) if not collected.get(slot)]
        proposed = {
            "action": "trigger_subculture_workflow",
            "workflow_name": "subculture",
            "source_message": payload.get("source_message"),
            "request_agent_run_id": payload.get("agent_run_id"),
        }
    else:
        collected = dict(payload.get("collected_fields") or {})
        missing = list(payload.get("missing_fields") or [])
        proposed = {
            "operation": payload.get("operation"),
            "tool_name": payload.get("tool_name"),
            "candidates": list(payload.get("candidates") or []),
            "source_message": payload.get("source_message"),
        }
        task_type = f"strain_{payload.get('operation') or 'mutation'}"
    try:
        conversation_tasks.create_task(
            conversation_id=conversation_id,
            owner=str(conversation["owner"]),
            workspace_id=str(conversation.get("workspace_id") or "shared"),
            goal_text=str(payload.get("source_message") or task_type),
            task_type=task_type,
            status="collecting" if missing else "ready",
            state_changing=True,
            collected_slots=collected,
            missing_slots=missing,
            proposed_action=proposed,
            latest_agent_run_id=payload.get("agent_run_id"),
            legacy_source=legacy_source,
            expires_at=payload.get("expires_at"),
        )
        return True
    except conversation_tasks.ActiveTaskConflict:
        logger.warning("Legacy task state conflicts with an active database task and was preserved: %s", path)
        return False


def import_legacy_task_states() -> dict[str, int]:
    imported = 0
    skipped = 0
    for directory, task_type in ((WORKFLOW_REQUEST_DIR, "subculture"), (PENDING_FORM_DIR, "pending_form")):
        root = Path(directory)
        if not root.exists():
            continue
        for path in root.glob("*.json"):
            payload = _read(path)
            if payload and _import(path, payload, task_type=task_type):
                imported += 1
            else:
                skipped += 1
    return {"imported": imported, "skipped": skipped}
