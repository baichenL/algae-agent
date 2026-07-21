# app/services/memory/memory_service.py 
# memory读取入口，负责管理各种类型的内存数据，包括状态内存、待办内存、会话内存和决策日志
# 不只依赖LLM上下文，还直接从数据库读取状态和待办列表，提供给Agent决策使用
from typing import Any, Dict, List

from app.core.database import (
    get_algae_status,
    get_recent_experiments as db_get_recent_experiments,
    list_algae_status,
    list_pending_actions,
)
from app.services.audit.agent_decision_log import log_agent_decision

# 从 DB 读取当前实验室整体品系状态，输出是品系状态列表
def get_status_memory() -> List[Dict[str, Any]]:
    """Return DB-first strain status memory."""
    strains = list_algae_status()
    try:
        from app.services.notifications.notification_service import attach_reminder_status_to_strains

        return attach_reminder_status_to_strains(strains)
    except Exception as exc:
        from app.services.observability.error_events import record_error_event

        record_error_event(
            layer="memory_db",
            component="memory_service",
            operation="attach_reminder_status_to_strains",
            severity="warning",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"strain_count": len(strains)},
        )
        return strains

# 从 DB 读取单个品系状态，输出是品系状态字典
def get_strain_memory(strain_id: str) -> Dict[str, Any] | None:
    """Return DB-first current state for one strain."""
    strain = get_algae_status(strain_id)
    if not strain:
        return None
    try:
        from app.services.notifications.notification_service import attach_reminder_status_to_strains

        return attach_reminder_status_to_strains([strain])[0]
    except Exception as exc:
        from app.services.observability.error_events import record_error_event

        record_error_event(
            layer="memory_db",
            component="memory_service",
            operation="attach_reminder_status_to_strain",
            severity="warning",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"strain_id": strain_id},
        )
        return strain

# 从 DB 读取所有待办审批列表，输出是pending actions列表
def get_pending_memory() -> List[Dict[str, Any]]:
    """Return pending approval memory."""
    return list_pending_actions()

# 从 DB 读取相关操作和目标品系的待办审批列表
def get_active_pending_actions(
    strain_id: str | None = None,
    action_type: str | None = None,
) -> List[Dict[str, Any]]:
    """Return active pending approvals, optionally filtered by strain and action."""
    actions = list_pending_actions(status="pending")
    filtered = []
    for item in actions:
        payload = item.get("payload") or {}
        data = payload.get("data") or {}
        payload_type = payload.get("type")
        if action_type and action_type not in {item.get("action_type"), payload_type}:
            continue
        if strain_id and data.get("strain_id") != strain_id:
            continue
        filtered.append(item)
    return filtered

# 从 DB 读取最近实验数据，输出是实验数据列表
def get_recent_experiments(
    strain_id: str | None = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return compact recent experiment memory without full hardware logs."""
    return db_get_recent_experiments(strain=strain_id, limit=limit)

# 读取当前会话历史
def get_session_memory(session_id: str) -> list:
    """Return per-session conversation memory."""
    from app.core.config import DEFAULT_SYSTEM_PROMPT, MAX_SESSION_MESSAGES, load_memory
    from app.core.db import assistant_conversations

    if assistant_conversations.get_conversation(session_id):
        messages = assistant_conversations.list_messages(session_id, limit=max(1, MAX_SESSION_MESSAGES - 1))
        return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + [
            {"role": item["role"], "content": item["content"]} for item in messages
        ]

    return load_memory(session_id)

# 保存当前会话历史
def save_session_memory(session_id: str, conversation_history: list) -> None:
    """Persist per-session conversation memory."""
    from app.core.config import save_memory
    from app.core.db import assistant_conversations

    if assistant_conversations.get_conversation(session_id):
        return

    save_memory(session_id, conversation_history)

# 追加一条决策日志到决策日志数据库中
def append_decision_event(event: Dict[str, Any]) -> None:
    """Persist lightweight agent decision memory."""
    normalized = {
        "agent_run_id": None,
        "session_id": None,
        "intent": None,
        "strain_id": None,
        "pending_id": None,
        "tool": None,
        **(event or {}),
    }
    log_agent_decision(normalized)
