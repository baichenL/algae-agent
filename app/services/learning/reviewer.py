from __future__ import annotations

from typing import Any

from app.core.db.agent_events import list_agent_run_events
from app.core.db.agent_events import record_agent_run_event
from app.services.learning.store import insert_learning_review, upsert_memory


LEARNING_EVENT_TYPES = {
    "agent_action_blocked",
    "agent_replan_directive_created",
    "approval_resume_rejected",
    "protocol_hash_mismatch",
    "stale_state_detected",
    "run_failed",
}


def _digest_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") or {}
    compact = {
        "event_type": event.get("event_type"),
        "layer": event.get("layer"),
        "step_index": payload.get("step_index"),
    }
    if event.get("event_type") == "agent_replan_directive_created":
        directive = payload.get("directive") or {}
        compact.update({
            "reason": directive.get("reason"),
            "blocked_by": directive.get("blocked_by"),
            "selection_reason": directive.get("selection_reason"),
        })
    elif event.get("event_type") == "agent_action_blocked":
        policy = payload.get("policy") or {}
        action = payload.get("action") or {}
        compact.update({
            "action_name": action.get("action_name"),
            "policy_category": policy.get("category"),
            "policy_reason": policy.get("reason"),
        })
    else:
        compact["payload"] = {
            key: value
            for key, value in payload.items()
            if key in {"status", "reason", "pending_id", "failure_stage", "failure_layer", "error_event_id", "validation"}
        }
    return compact


def build_run_digest(agent_run_id: str) -> dict[str, Any]:
    events = list_agent_run_events(agent_run_id)
    triggers = [event for event in events if event.get("event_type") in LEARNING_EVENT_TYPES]
    return {
        "agent_run_id": agent_run_id,
        "trigger_count": len(triggers),
        "triggers": [_digest_event(event) for event in triggers[:10]],
        "event_count": len(events),
    }


def _candidate_from_digest(digest: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    triggers = digest.get("triggers") or []
    for item in triggers:
        event_type = item.get("event_type")
        if event_type == "agent_action_blocked":
            return (
                "memory_candidate",
                {
                    "scope": "agent",
                    "content": (
                        "When policy blocks an action, explain the approval or safety boundary "
                        "instead of retrying the same tool call."
                    ),
                    "confidence": 0.7,
                },
                "accepted",
            )
        if event_type in {"approval_resume_rejected", "stale_state_detected", "protocol_hash_mismatch"}:
            reason = (item.get("payload") or {}).get("reason") or "stale_or_invalid_approval"
            return (
                "memory_candidate",
                {
                    "scope": "agent",
                    "content": (
                        f"When approval resume is rejected because {reason}, do not execute; "
                        "explain stale state and ask for a fresh approval request."
                    ),
                    "confidence": 0.75,
                },
                "accepted",
            )
        if event_type == "agent_replan_directive_created" and item.get("reason") not in {"no_loop_plan", "loop_plan_complete"}:
            return (
                "memory_candidate",
                {
                    "scope": "agent",
                    "content": (
                        "When observation-driven replan is needed, prefer deterministic DB-backed "
                        "targets and stop at pending approval for high-risk workflows."
                    ),
                    "confidence": 0.65,
                },
                "accepted",
            )
        if event_type == "run_failed":
            return (
                "memory_candidate",
                {
                    "scope": "agent",
                    "content": "When a run fails, preserve the error boundary in trace and avoid claiming tool success.",
                    "confidence": 0.6,
                },
                "accepted",
            )
    return ("no_learning_needed", {"reason": "no eligible learning trigger"}, "accepted")


def _validate_candidate(candidate_type: str, candidate: dict[str, Any]) -> tuple[bool, str]:
    if candidate_type == "no_learning_needed":
        return True, "no_learning_needed"
    if candidate_type != "memory_candidate":
        return False, "skill_candidates_require_manual_review_v1"
    content = str(candidate.get("content") or "")
    if not 20 <= len(content) <= 500:
        return False, "memory_length_out_of_bounds"
    lowered = content.casefold()
    forbidden = ("approve pending", "bypass", "directly execute", "trigger hardware", "ignore policy")
    if any(term in lowered for term in forbidden):
        return False, "candidate_violates_safety_boundary"
    return True, "accepted_low_risk_memory"


def run_post_run_learning_review(
    *,
    agent_run_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    digest = build_run_digest(agent_run_id)
    if not digest.get("trigger_count"):
        return {"status": "skipped", "reason": "no_learning_trigger", "run_digest": digest}

    candidate_type, candidate, initial_status = _candidate_from_digest(digest)
    valid, reason = _validate_candidate(candidate_type, candidate)
    status = initial_status if valid else "pending_review"
    applied_memory_id = None
    if valid and candidate_type == "memory_candidate":
        applied_memory_id = upsert_memory(
            scope=str(candidate.get("scope") or "agent"),
            content=str(candidate.get("content") or ""),
            source_run_id=agent_run_id,
            confidence=float(candidate.get("confidence") or 0.5),
            status="active",
        )

    review_id = insert_learning_review(
        agent_run_id=agent_run_id,
        session_id=session_id,
        review_type="post_run",
        candidate_type=candidate_type,
        candidate={**candidate, "applied_memory_id": applied_memory_id},
        run_digest=digest,
        status=status,
        validation_reason=reason,
    )
    record_agent_run_event(
        agent_run_id=agent_run_id,
        session_id=session_id,
        event_type="agent_learning_review_completed",
        layer="agent_learning",
        payload={
            "review_id": review_id,
            "candidate_type": candidate_type,
            "status": status,
            "validation_reason": reason,
            "applied_memory_id": applied_memory_id,
            "run_digest": digest,
        },
    )
    return {
        "status": status,
        "review_id": review_id,
        "candidate_type": candidate_type,
        "validation_reason": reason,
        "applied_memory_id": applied_memory_id,
    }
