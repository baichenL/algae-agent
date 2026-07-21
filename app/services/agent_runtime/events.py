from __future__ import annotations

import uuid
from typing import Any

from app.core.db.agent_events import (
    finish_agent_run as _finish_agent_run,
    mark_agent_run_resumed as _mark_agent_run_resumed,
    mark_agent_run_waiting_approval as _mark_agent_run_waiting_approval,
    record_agent_run_event as _record_agent_run_event,
    start_agent_run as _start_agent_run,
)
from app.services.agent_runtime.state import AgentRunState


def start_run(
    session_id: str,
    user_message: str,
    agent_run_id: str | None = None,
    *,
    graph_thread_id: str | None = None,
    graph_definition_version: str | None = None,
    state_schema_version: int | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    attempt_no: int = 1,
) -> str:
    agent_run_id = agent_run_id or str(uuid.uuid4())
    return _start_agent_run(
        agent_run_id=agent_run_id,
        session_id=session_id,
        user_message=user_message,
        graph_thread_id=graph_thread_id,
        graph_definition_version=graph_definition_version,
        state_schema_version=state_schema_version,
        conversation_id=conversation_id or session_id,
        task_id=task_id,
        attempt_no=attempt_no,
    )

# 作用是记录一个运行事件，通常用于跟踪代理运行的状态和行为。它接受代理运行的ID、会话ID、事件类型、层级以及可选的负载数据，并将这些信息传递给底层的记录函数 `_record_agent_run_event`，以便在数据库中存储该事件。
def record_run_event(
    agent_run_id: str | None,
    *,
    session_id: str | None = None,
    event_type: str,
    layer: str,
    payload: dict[str, Any] | None = None,
) -> int | None:
    return _record_agent_run_event(
        agent_run_id=agent_run_id,
        session_id=session_id,
        event_type=event_type,
        layer=layer,
        payload=payload,
    )

# 作用是记录一个代理事件，通常用于跟踪代理运行的状态和行为。它接受代理运行的状态对象、事件类型、层级以及可选的负载数据，并将这些信息传递给底层的记录函数 `record_run_event`，以便在数据库中存储该事件。
def record_agent_event(
    state: AgentRunState,
    event_type: str,
    layer: str,
    payload: dict[str, Any] | None = None,
) -> int | None:
    event_payload = {
        "step_index": state.step_index,
        **(payload or {}),
    }
    return record_run_event(
        state.agent_run_id,
        session_id=state.session_id,
        event_type=event_type,
        layer=layer,
        payload=event_payload,
    )


def finish_run(
    agent_run_id: str | None,
    *,
    status: str,
    final_route: str | None = None,
    risk_level: str | None = None,
    response_summary: str | None = None,
    error_event_id: int | None = None,
) -> None:
    _finish_agent_run(
        agent_run_id=agent_run_id,
        status=status,
        final_route=final_route,
        risk_level=risk_level,
        response_summary=response_summary,
        error_event_id=error_event_id,
    )


def mark_run_waiting_approval(
    *,
    agent_run_id: str,
    pending_id: int,
    graph_thread_id: str,
    last_node: str = "AwaitApproval",
) -> bool:
    return _mark_agent_run_waiting_approval(
        agent_run_id=agent_run_id,
        pending_id=pending_id,
        graph_thread_id=graph_thread_id,
        last_node=last_node,
    )


def mark_run_resumed(
    *,
    agent_run_id: str,
    graph_thread_id: str,
    last_node: str = "LoadApproval",
) -> None:
    _mark_agent_run_resumed(
        agent_run_id=agent_run_id,
        graph_thread_id=graph_thread_id,
        last_node=last_node,
    )
