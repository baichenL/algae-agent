from __future__ import annotations

from typing import Any

from app.core.db.agent_events import (
    get_agent_run,
    list_agent_error_events,
    list_agent_run_events,
    list_agent_runs,
)
from app.services.learning.store import list_agent_learning_reviews


def list_agent_run_summaries(
    *,
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return list_agent_runs(session_id=session_id, status=status, limit=limit)


def build_agent_run_trace(agent_run_id: str) -> dict[str, Any] | None:
    run = get_agent_run(agent_run_id)
    if not run:
        return None

    events = list_agent_run_events(agent_run_id)
    errors = list_agent_error_events(agent_run_id)
    trace, warnings = _rebuild_trace(events, errors)
    learning_reviews = list_agent_learning_reviews(agent_run_id=agent_run_id, limit=20)
    trace["learning"] = _learning_trace(events, learning_reviews)
    if warnings:
        trace["warnings"] = warnings
    return {
        "status": "success",
        "run": run,
        "trace": trace,
        "raw_events": [_public_event(item) for item in events],
    }


def _public_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": event.get("id"),
        "event_type": event.get("event_type"),
        "layer": event.get("layer"),
        "payload": event.get("payload") or {},
        "created_at": event.get("created_at"),
    }


def _rebuild_trace(events: list[dict[str, Any]], errors: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    steps: dict[int, dict[str, Any]] = {}
    context_payloads: list[dict[str, Any]] = []
    lifecycle_events: list[dict[str, Any]] = []
    plan_payload = None
    stop_reason = None
    final_status = None
    failure_stage = None
    failure_layer = None
    error_event_id = None
    learning_events: list[dict[str, Any]] = []
    model_inputs: list[dict[str, Any]] = []
    status_bars: list[dict[str, Any]] = []
    context_budget_errors: list[dict[str, Any]] = []

    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        step_index = int(payload.get("step_index") or 0)

        if event_type in {"run_started", "run_finished", "run_failed"}:
            lifecycle_events.append(_public_event(event))
            if event_type == "run_failed":
                final_status = "failed"
                failure_layer = event.get("layer") or failure_layer
                error_event_id = payload.get("error_event_id") or error_event_id
                failure_stage = payload.get("failure_stage") or payload.get("stage") or failure_stage
            elif event_type == "run_finished":
                final_status = payload.get("status") or final_status

        if event_type and event_type.startswith("agent_learning_"):
            learning_events.append(_public_event(event))

        if event_type == "model_input_used":
            model_inputs.append(payload)
        elif event_type == "agent_status_bar_updated":
            status_bars.append({"step_index": step_index, **payload})
        elif event_type == "context_budget_exceeded":
            context_budget_errors.append(payload)

        if event_type == "context_built":
            context_payloads.append(payload)
            continue
        if event_type == "agent_plan_created":
            plan_payload = payload.get("plan") or payload
            continue

        if step_index:
            step = steps.setdefault(step_index, {"step_index": step_index})
        else:
            step = None

        if event_type == "agent_decision_made" and step is not None:
            step["decision"] = payload.get("decision") or {}
        elif event_type == "agent_policy_evaluated" and step is not None:
            step["action"] = payload.get("action") or {}
            step["policy_verdict"] = payload.get("policy") or {}
        elif event_type == "agent_observation_collected" and step is not None:
            step["observation"] = payload.get("observation") or {}
        elif event_type == "agent_observation_assessed" and step is not None:
            step["observation_assessment"] = payload.get("assessment") or {}
        elif event_type == "agent_replan_started" and step is not None:
            step["replan_started"] = {
                "loop_mode": payload.get("loop_mode"),
                "loop_plan_index": payload.get("loop_plan_index"),
            }
        elif event_type == "agent_replan_decision_made" and step is not None:
            step["replan_decision"] = payload.get("decision") or {}
            step["replan_source"] = payload.get("replan_source") or "none"
        elif event_type == "agent_replan_blocked" and step is not None:
            step["replan_blocked"] = payload.get("directive") or payload
            step["replan_source"] = (payload.get("directive") or {}).get("replan_source") or "none"
        elif event_type == "agent_replan_directive_created" and step is not None:
            step["replan_directive"] = payload.get("directive") or {}
            step["replan_source"] = (payload.get("directive") or {}).get("replan_source") or "none"
        elif event_type == "agent_step_finished" and step is not None:
            finished = payload.get("step") or {}
            step.setdefault("plan_step", finished.get("plan_step") or {})
            step.setdefault("decision", finished.get("decision") or {})
            step.setdefault("action", finished.get("action") or {})
            step.setdefault("policy_verdict", finished.get("policy_verdict") or {})
            step.setdefault("observation", finished.get("observation") or {})
            step.setdefault("observation_assessment", finished.get("observation_assessment") or {})
            step.setdefault("replan_directive", finished.get("replan_directive") or {})
            step.setdefault(
                "replan_source",
                (finished.get("replan_directive") or {}).get("replan_source") or "none",
            )
            step["terminal_status"] = finished.get("terminal_status")
        elif event_type == "agent_loop_stopped":
            directive = payload.get("directive") or {}
            stop_reason = directive.get("reason")
            final_status = directive.get("terminal_status")
        elif event_type == "run_finished":
            final_status = payload.get("status") or final_status

    ordered_steps = [steps[index] for index in sorted(steps)]
    for step in ordered_steps:
        missing = [
            key
            for key in ("decision", "policy_verdict", "action", "observation")
            if key not in step
        ]
        if missing:
            warnings.append(f"step_{step['step_index']}_missing_{','.join(missing)}")

    if not ordered_steps:
        warnings.append("no_agent_steps_found")

    public_errors = [_public_error(item) for item in errors]
    if public_errors:
        latest_error = public_errors[-1]
        failure_stage = (
            failure_stage
            or latest_error.get("operation")
            or latest_error.get("component")
            or latest_error.get("layer")
        )
        failure_layer = failure_layer or latest_error.get("layer")
        error_event_id = error_event_id or latest_error.get("id")
        final_status = final_status or "failed"

    context = _merge_context(context_payloads)
    return (
        {
            "context": context,
            "plan": plan_payload or {},
            "steps": ordered_steps,
            "stop_reason": stop_reason,
            "final_status": final_status,
            "failure_stage": failure_stage,
            "failure_layer": failure_layer,
            "error_event_id": error_event_id,
            "lifecycle_events": lifecycle_events,
            "errors": public_errors,
            "learning_events": learning_events,
            "model_inputs": model_inputs,
            "status_bars": status_bars,
            "context_budget_errors": context_budget_errors,
        },
        warnings,
    )


def _learning_trace(events: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    selected_contexts = []
    for event in events:
        if event.get("event_type") != "context_built":
            continue
        payload = event.get("payload") or {}
        context_bundle = payload.get("context_bundle") or {}
        policy_payload = (context_bundle.get("policy_context") or {}).get("payload") or {}
        learning_context = policy_payload.get("learning_context") or {}
        if learning_context:
            selected_contexts.append(learning_context)
    return {
        "selected_contexts": selected_contexts,
        "reviews": reviews,
    }


def _merge_context(context_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not context_payloads:
        return {}
    latest = context_payloads[-1]
    context_bundle = latest.get("context_bundle")
    if isinstance(context_bundle, dict) and context_bundle:
        return {
            **context_bundle,
            "raw_context_events": context_payloads,
        }
    return {
        "fact_context": {
            "strain_count": latest.get("strain_count"),
            "pending_count": latest.get("pending_count"),
            "recent_experiment_count": latest.get("recent_experiment_count"),
            "target_pending_count": latest.get("target_pending_count"),
        },
        "session_context": {
            "session_id": latest.get("session_id"),
            "runtime": (latest.get("runtime_context") or {}),
        },
        "tool_context": {
            "previous_observation_count": (latest.get("runtime_context") or {}).get(
                "previous_observation_count"
            )
        },
        "policy_context": {
            "policy_count": (latest.get("runtime_context") or {}).get("policy_count")
        },
        "raw_context_events": context_payloads,
    }


def _public_error(error: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": error.get("id"),
        "layer": error.get("layer"),
        "component": error.get("component"),
        "operation": error.get("operation"),
        "severity": error.get("severity"),
        "error_type": error.get("error_type"),
        "error_message": error.get("error_message"),
        "metadata": error.get("metadata") or {},
        "created_at": error.get("created_at"),
    }
