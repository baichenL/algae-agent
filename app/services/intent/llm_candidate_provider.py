from __future__ import annotations

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.core.config import client
from app.core.model_registry import model_name
from app.services.context import (
    ContextBudgetExceeded,
    ModelCallTimer,
    assemble_model_input,
    context_input_mode,
    model_call_metrics,
)
from app.services.intent.routing_models import EffectKind, IntentFrame, RiskLevel, RouteKind, SpeechAct
from app.services.skills import skill_manifest_for_router


@dataclass(frozen=True)
class LlmRouteCandidate:
    route_kind: RouteKind
    confidence: float
    speech_act: SpeechAct | None = None
    # Legacy constructor compatibility only. Host-side routing ignores this
    # value and derives risk from the selected route/tool metadata.
    risk_level: RiskLevel | None = None
    evidence: tuple[str, ...] = field(default_factory=tuple)
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    mentioned_targets: tuple[str, ...] = field(default_factory=tuple)
    source: str = "llm"


_LAST_ROUTER_DIAGNOSTICS: ContextVar[dict[str, Any] | None] = ContextVar(
    "last_router_diagnostics",
    default=None,
)


def intent_router_mode() -> str:
    configured = os.getenv("INTENT_ROUTER_MODE", "").strip().lower()
    if configured in {"rules", "shadow", "hybrid"}:
        return configured
    legacy = os.getenv("HYBRID_ROUTER_LLM_ENABLED", "").strip().lower()
    return "hybrid" if legacy in {"1", "true", "yes"} else "rules"


def current_router_diagnostics() -> dict[str, Any] | None:
    value = _LAST_ROUTER_DIAGNOSTICS.get()
    return dict(value) if value else None


def _context_summary(context_snapshot: Any) -> dict[str, Any]:
    strains = list(getattr(context_snapshot, "strains", []) or [])
    pending = list(getattr(context_snapshot, "pending_actions", []) or [])
    return {
        "strain_ids": [item.get("strain_id") for item in strains[:20] if isinstance(item, dict)],
        "pending_count": len(pending),
        "pending_types": [item.get("action_type") for item in pending[:10] if isinstance(item, dict)],
    }


def _parse_intent_frame(raw: str) -> IntentFrame | None:
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    # One-release compatibility for the former candidate-array response.
    if isinstance(parsed.get("candidates"), list) and parsed["candidates"]:
        item = parsed["candidates"][0]
        if not isinstance(item, dict):
            return None
        parsed = {
            "schema_version": "intent-frame/v1",
            "goal": str(item.get("evidence", ["semantic routing"])[0]),
            "primary_route": item.get("route_kind"),
            "alternatives": [],
            "speech_act": item.get("speech_act") or "query",
            "requested_effect": "propose" if item.get("route_kind") in {"workflow", "write_action", "scientific_task"} else "read",
            "entity_mentions": item.get("mentioned_targets") or [],
            "slots": {},
            "missing_slots": item.get("missing_fields") or [],
            "suggested_skill": None,
            "suggested_tools": [],
            "confidence": float(item.get("confidence") or 0.0),
            "evidence": item.get("evidence") or [],
        }
    try:
        return IntentFrame.model_validate(parsed)
    except (ValidationError, TypeError, ValueError):
        return None


def _remote_intent_frame(
    *,
    original_text: str,
    normalized_text: str,
    context_snapshot: Any,
    active_pending_form: dict[str, Any] | None,
    active_workflow_request: dict[str, Any] | None,
) -> IntentFrame | None:
    mode = intent_router_mode()
    if mode == "rules":
        _LAST_ROUTER_DIAGNOSTICS.set({"mode": mode, "status": "disabled"})
        return None
    dynamic_request = {
        "normalized_text": normalized_text,
        "active_pending_form": bool(active_pending_form),
        "active_workflow_request": bool(active_workflow_request),
    }
    task_protocol = {
        "task": "Return one semantic IntentFrame. Never decide risk, permission, approval, execution, or database entity existence.",
        "routes": sorted(item.value for item in RouteKind),
        "speech_acts": sorted(item.value for item in SpeechAct),
        "requested_effects": sorted(item.value for item in EffectKind),
        "output_schema": {
            "schema_version": "intent-frame/v1",
            "goal": "short normalized user goal",
            "primary_route": "workflow",
            "alternatives": [],
            "speech_act": "command",
            "requested_effect": "propose",
            "entity_mentions": ["raw mentions only"],
            "slots": {},
            "missing_slots": [],
            "suggested_skill": "subculture",
            "suggested_tools": [],
            "confidence": 0.0,
            "evidence": ["short semantic evidence"],
        },
    }
    envelope = None
    try:
        if context_input_mode() == "legacy":
            prompt = {**task_protocol, **dynamic_request, "context_summary": _context_summary(context_snapshot), "skills": skill_manifest_for_router()}
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative route candidate generator for a lab agent. "
                        "Return strict JSON only. Never authorize execution."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ]
        else:
            envelope = assemble_model_input(
                request_kind="router",
                route_kind=None,
                current_user_message=original_text,
                context_snapshot=context_snapshot,
                system_instruction=(
                    "You are a conservative route candidate generator for a lab agent. "
                    "Return strict JSON only. Never authorize execution."
                ),
                task_protocol=task_protocol,
                extra_context=dynamic_request,
                entity_mentions=[],
            )
            messages = envelope.to_messages(original_text)
        timer = ModelCallTimer()
        response = client.chat.completions.create(
            model=model_name("router"),
            messages=messages,
            temperature=0,
        )
        frame = _parse_intent_frame(response.choices[0].message.content or "")
        _LAST_ROUTER_DIAGNOSTICS.set({
            "mode": mode,
            "status": "parsed" if frame else "invalid_json",
            "intent_frame": frame.model_dump(mode="json") if frame else None,
            "model_input": model_call_metrics(envelope, response=response, elapsed_ms=timer.elapsed_ms()) if envelope else None,
        })
        return frame
    except ContextBudgetExceeded as exc:
        _LAST_ROUTER_DIAGNOSTICS.set({
            "mode": mode,
            "status": "context_budget_exceeded",
            "budget_report": exc.report.to_dict(),
        })
        return None
    except Exception as exc:
        _LAST_ROUTER_DIAGNOSTICS.set({
            "mode": mode,
            "status": "fallback",
            "error": exc.__class__.__name__,
            "model_input": model_call_metrics(envelope, elapsed_ms=timer.elapsed_ms(), error=exc) if envelope and "timer" in locals() else None,
        })
        return None


def collect_llm_route_candidates(
    *,
    original_text: str,
    normalized_text: str,
    context_snapshot: Any,
    active_pending_form: dict[str, Any] | None = None,
    active_workflow_request: dict[str, Any] | None = None,
) -> list[LlmRouteCandidate]:
    frame = _remote_intent_frame(
        original_text=original_text,
        normalized_text=normalized_text,
        context_snapshot=context_snapshot,
        active_pending_form=active_pending_form,
        active_workflow_request=active_workflow_request,
    )
    if not frame or intent_router_mode() != "hybrid":
        return []
    routes = (frame.primary_route, *frame.alternatives[:4])
    return [
        LlmRouteCandidate(
            route_kind=route,
            confidence=frame.confidence if index == 0 else max(0.0, frame.confidence - 0.1),
            speech_act=frame.speech_act,
            evidence=frame.evidence or ("intent_frame",),
            missing_fields=frame.missing_slots,
            mentioned_targets=frame.entity_mentions,
        )
        for index, route in enumerate(dict.fromkeys(routes))
    ]
