from __future__ import annotations

import time
from typing import Any

from app.services.observability.error_events import record_error_event
from app.services.user_memory.extractor import auto_write_threshold, extract_candidates, extraction_enabled, shadow_mode
from app.services.user_memory.store import apply_memory, create_candidate, get_settings, record_memory_event


def process_completed_turn(*, owner_id: str, workspace_id: str, conversation_id: str,
                           user_message_id: int, user_text: str, assistant_text: str,
                           source_run_id: str | None = None) -> dict[str, Any]:
    if not extraction_enabled():
        return {"status": "skipped", "reason": "feature_disabled"}
    settings = get_settings(owner_id, workspace_id)
    if not settings.get("enabled"):
        return {"status": "skipped", "reason": "user_not_opted_in"}
    try:
        started = time.perf_counter()
        model, extracted = extract_candidates(user_text, assistant_text)
        if model == "deterministic_sensitive_filter":
            record_memory_event(
                owner_id=owner_id, workspace_id=workspace_id,
                event_type="candidate_blocked", actor="memory_safety_filter",
                reason="prohibited_sensitive_input",
                metadata={"source_message_id": user_message_id},
            )
            return {"status": "blocked", "reason": "prohibited_sensitive_input"}
        results = []
        for item in extracted:
            should_auto_write = (
                settings.get("auto_write_low_risk")
                and not shadow_mode()
                and item["sensitivity"] == "low"
                and float(item["confidence"]) >= auto_write_threshold()
            )
            candidate = create_candidate(
                owner_id=owner_id, workspace_id=workspace_id, model_name=model,
                source_conversation_id=conversation_id, source_message_id=user_message_id,
                source_run_id=source_run_id, status="accepted" if should_auto_write else "pending", **item,
            )
            memory = None
            action = None
            if should_auto_write:
                action, memory = apply_memory(
                    owner_id=owner_id, workspace_id=workspace_id, source_conversation_id=conversation_id,
                    source_message_id=user_message_id, source_run_id=source_run_id,
                    actor="memory_extractor", **{key: item[key] for key in ("memory_type", "predicate", "value", "confidence", "sensitivity", "reason")},
                )
                from app.core.db.connection import connect
                from app.core.time_utils import local_time_string
                with connect() as conn:
                    conn.execute(
                        "UPDATE user_memory_candidates SET applied_memory_id = ?, updated_at = ? WHERE id = ?",
                        (memory["id"], local_time_string(), candidate["id"]),
                    )
                    conn.commit()
            results.append({"candidate_id": candidate.get("id"), "action": action, "memory_id": (memory or {}).get("id")})
        record_memory_event(
            owner_id=owner_id, workspace_id=workspace_id,
            event_type="extraction_completed", actor="memory_extractor",
            metadata={"candidate_count": len(results), "latency_ms": round((time.perf_counter() - started) * 1000, 2)},
        )
        return {"status": "success", "candidate_count": len(results), "results": results}
    except Exception as exc:
        try:
            record_error_event(
                session_id=conversation_id,
                layer="user_memory",
                component="completed_turn_processor",
                operation="extract_user_memory",
                severity="warning",
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={"owner_id": owner_id, "workspace_id": workspace_id, "source_message_id": user_message_id},
            )
        except Exception:
            pass
        return {"status": "failed", "reason": "memory_extraction_failed"}
