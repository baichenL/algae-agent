from __future__ import annotations

from typing import Any


_OUTCOME_MAP = {
    "success": "success",
    "succeeded": "success",
    "partial": "partial",
    "needs_more_info": "needs_input",
    "needs_clarification": "needs_input",
    "pending": "pending",
    "waiting_approval": "pending",
    "paused": "paused",
    "error": "failed",
    "failed": "failed",
    "blocked": "failed",
}


def _references(output: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for kind, key in (
        ("agent_trace", "trace_id"),
        ("scientific_run", "scientific_run_id"),
        ("pending", "pending_id"),
        ("task", "task_id"),
    ):
        value = output.get(key)
        if value is not None:
            refs.append({"type": kind, "id": value})
    for item in output.get("references") or []:
        if isinstance(item, dict) and item not in refs:
            refs.append(item)
    return refs


def _presentation_blocks(
    output: dict[str, Any],
    *,
    outcome_status: str,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    draft = output.get("draft")
    if isinstance(draft, dict):
        blocks.append({
            "type": "email_draft",
            "draft": draft,
            "target": output.get("target")
            or (output.get("email_request_spec") or {}).get("target"),
            "send": False,
        })
    if output.get("pending_id") is not None:
        blocks.append({
            "type": "approval",
            "pending_id": output.get("pending_id"),
            "status": "pending",
            "executable": False,
            "requires_human_approval": True,
        })
    candidates = list(output.get("candidate_plans") or [])
    simulations = list(output.get("simulation_results") or [])
    plan_patches = list(output.get("plan_patches") or [])
    if candidates or simulations or output.get("scientific_run_id"):
        blocks.append({
            "type": "scientific_result",
            "candidate_count": len(candidates)
            or int((output.get("completed_work") or {}).get("candidate_count") or 0),
            "candidates": candidates,
            "validations": list(output.get("validations") or []),
            "simulations": simulations,
            "plan_patches": plan_patches,
            "comparison": output.get("comparison") or {},
            "scientific_run_id": output.get("scientific_run_id"),
        })
    if outcome_status == "paused":
        blocks.append({
            "type": "paused_task",
            "task_id": output.get("task_id"),
            "reason": output.get("pause_reason") or output.get("reason"),
            "completed_work": output.get("completed_work") or {},
            "remaining_work": list(output.get("remaining_work") or []),
            "budget": output.get("budget") or {},
        })
    return blocks


def build_answer_envelope(
    *,
    natural_reply: str,
    output: dict[str, Any],
) -> dict[str, Any]:
    status = str(output.get("outcome_status") or output.get("status") or "success")
    outcome_status = _OUTCOME_MAP.get(status, status if status in _OUTCOME_MAP.values() else "success")
    direct_answer = str(output.get("direct_answer") or natural_reply or "")
    source_statuses = list(output.get("source_statuses") or [])
    unknowns = list(output.get("unknowns") or [])
    if not unknowns:
        unknowns = [
            str(item.get("impact"))
            for item in source_statuses
            if item.get("status") in {"empty", "failed", "degraded"} and item.get("impact")
        ]
    next_actions = list(output.get("next_actions") or [])
    if outcome_status == "paused" and not next_actions:
        next_actions.append({
            "action": "resume_task",
            "label": "继续任务",
            "task_id": output.get("task_id"),
        })
    if outcome_status == "pending" and output.get("pending_id") is not None and not next_actions:
        next_actions.append({
            "action": "review_pending",
            "label": "查看审批",
            "pending_id": output.get("pending_id"),
        })
    return {
        "schema_version": "answer-envelope/v2",
        "outcome_status": outcome_status,
        "direct_answer": direct_answer,
        "confirmed_facts": list(output.get("confirmed_facts") or []),
        "inferences": list(output.get("inferences") or []),
        "simulation_results": list(output.get("simulation_results") or []),
        "recommendations": list(output.get("recommendations") or []),
        "unknowns": unknowns,
        "source_statuses": source_statuses,
        "completed_work": output.get("completed_work") or {},
        "remaining_work": list(output.get("remaining_work") or []),
        "budget": output.get("budget") or {},
        "references": _references(output),
        "next_actions": next_actions,
        "presentation_blocks": _presentation_blocks(
            output,
            outcome_status=outcome_status,
        ),
    }


def attach_answer_envelope(response: Any) -> Any:
    output = dict(getattr(response, "agent_output", None) or {})
    envelope = build_answer_envelope(
        natural_reply=str(getattr(response, "natural_reply", "") or ""),
        output=output,
    )
    output["outcome_status"] = envelope["outcome_status"]
    output["answer_envelope"] = envelope
    response.agent_output = output
    return response
