import datetime
import json
import os
from typing import Any, Dict


DECISION_LOG_PATH = "data/outputs/agent_decision_log.jsonl"


def log_agent_decision(event: Dict[str, Any]) -> None:
    """Append a lightweight agent routing/tool decision event to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(DECISION_LOG_PATH), exist_ok=True)
        record = {
            "created_at": datetime.datetime.utcnow().isoformat(),
            **event,
        }
        with open(DECISION_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        from app.services.observability.error_events import record_error_event

        record_error_event(
            session_id=(event or {}).get("session_id"),
            agent_run_id=(event or {}).get("agent_run_id"),
            layer="memory_db",
            component="agent_decision_log",
            operation="log_agent_decision",
            severity="warning",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"path": DECISION_LOG_PATH, "event": event},
        )
        # Decision logging must never break the chat path.
        pass
