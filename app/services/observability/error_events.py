from __future__ import annotations

from typing import Any

from app.core.db.agent_events import record_error_event as _record_error_event


def record_error_event(
    *,
    session_id: str | None = None,
    agent_run_id: str | None = None,
    layer: str,
    component: str,
    operation: str,
    severity: str = "error",
    error_type: str | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    return _record_error_event(
        session_id=session_id,
        agent_run_id=agent_run_id,
        layer=layer,
        component=component,
        operation=operation,
        severity=severity,
        error_type=error_type,
        error_message=error_message,
        metadata=metadata,
    )
