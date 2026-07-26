
import re
import datetime
import os
from typing import Any, Dict

from app.core.db import pending_actions
from app.core.effects import stable_hash
from app.core.time_utils import local_now, local_time_string
from app.services.agent_runtime.safety_v2 import POLICY_VERSION
from app.services.context.context_builder import build_context_snapshot
from app.services.email.recipient_policy import get_default_recipients
from app.services.email.email_template import build_manual_check_draft
from app.services.intent.input_normalizer import normalize_text_value
from app.tools.registry_core import register_agent_tool
from app.tools.tool_schemas import EMAIL_DRAFT_SCHEMA


DIRECT_SEND_KEYWORDS = ["\u7acb\u5373\u53d1\u9001", "\u76f4\u63a5\u53d1\u9001", "\u73b0\u5728\u53d1\u9001", "\u9a6c\u4e0a\u53d1\u9001"]


def create_email_draft_from_user_message(message: str, context_snapshot: Any):
    recipients = get_default_recipients()
    metadata = {"strain_id": _extract_target_strain(message, context_snapshot)}
    return build_manual_check_draft(message, recipients, metadata)


def has_direct_send_intent(message: str) -> bool:
    text = normalize_text_value(message).casefold()
    return any(normalize_text_value(keyword).casefold() in text for keyword in DIRECT_SEND_KEYWORDS)


def maybe_send_email_from_frontend_intent(message: str, context_snapshot: Any):
    # An LLM-callable tool can only create an immutable draft. SMTP capability
    # exists exclusively in the trusted worker.
    return create_email_draft_from_user_message(message, context_snapshot)


def create_email_pending_from_draft(
    draft_payload: dict[str, Any],
    *,
    agent_run_id: str | None,
    created_by_tool_call_id: str,
    requester: str = "agent_tool",
    request_key: str | None = None,
) -> dict[str, Any]:
    """Create an approval-gated send proposal from the exact reviewed draft."""
    expires_at = local_time_string(
        local_now()
        + datetime.timedelta(
            seconds=max(60, int(os.getenv("PENDING_DEFAULT_TTL_SECONDS", "86400")))
        )
    )
    draft_hash = stable_hash(draft_payload)
    request_hash = stable_hash(request_key) if request_key else draft_hash
    envelope = {
        "proposal_version": 2,
        "target": {
            "type": "email_message",
            "recipients": list(draft_payload.get("recipients") or []),
        },
        "design_or_protocol": draft_payload,
        "input_resource_versions": {"draft_hash": draft_hash},
        "evidence_refs": [],
        "hypothesis_refs": [],
        "validation_result": {
            "valid": bool(draft_payload.get("recipients")),
            "recipient_count": len(draft_payload.get("recipients") or []),
        },
        "simulation_result": {"status": "not_applicable"},
        "policy_version": POLICY_VERSION,
        "validator_version": "email-draft-validator-v2",
        "simulator_version": "not_applicable",
        "risk_summary": {
            "external_effect": "formal_email_send",
            "approval_required": True,
        },
        "uncertainties": [],
        "expected_resource_versions": {"draft_hash": draft_hash},
        "expires_at": expires_at,
        "created_by_agent_run_id": agent_run_id,
        "created_by_tool_call_id": created_by_tool_call_id,
    }
    proposal_hash = stable_hash(envelope)
    pending_id = pending_actions.insert_pending_action(
        "email_send",
        {
            "type": "email_send",
            "data": {
                "draft": draft_payload,
                "draft_hash": draft_hash,
                "proposal_hash": proposal_hash,
            },
        },
        requester=requester,
        risk_level="medium",
        source="email_tool",
        agent_run_id=agent_run_id,
        execution_idempotency_key=f"email-request:{request_hash}",
        domain_dedupe_key=f"email-request:{request_hash}",
        expires_at=expires_at,
        proposal_version=2,
        proposal_hash=proposal_hash,
        proposal_envelope=envelope,
        policy_version=POLICY_VERSION,
        expected_resource_versions=envelope["expected_resource_versions"],
        created_by_tool_call_id=created_by_tool_call_id,
    )
    return {
        "pending_id": pending_id,
        "proposal_hash": proposal_hash,
        "expires_at": expires_at,
    }


@register_agent_tool(
    name="email_draft",
    schema=EMAIL_DRAFT_SCHEMA,
    risk_level="medium",
    effect_kind="draft",
    effect_class="proposal_write",
    side_effect="email_draft",
    requires_approval=True,
    allowed_callers=["chat_runtime", "frontend_form"],
    exposed_to_llm=True,
    executor_kind="artifact",
    audit_event_type="email_draft_created",
)
def handle_email_draft_tool(function_args: Dict[str, Any]) -> Dict[str, Any]:
    message = function_args.get("message", "")
    session_id = function_args.get("session_id", "default_session")
    context_snapshot = build_context_snapshot(session_id)

    try:
        result = maybe_send_email_from_frontend_intent(message, context_snapshot)
        draft = result
        draft_payload = draft.model_dump()
        pending_id = None
        proposal_hash = None
        expires_at = None
        if (
            function_args.get("created_by_tool_call_id")
            and function_args.get("request_approval")
        ):
            pending = create_email_pending_from_draft(
                draft_payload,
                agent_run_id=function_args.get("agent_run_id"),
                created_by_tool_call_id=str(function_args["created_by_tool_call_id"]),
                request_key=(
                    f"{session_id}:"
                    f"{normalize_text_value(function_args.get('source_message') or message).casefold()}"
                ),
            )
            pending_id = pending["pending_id"]
            proposal_hash = pending["proposal_hash"]
            expires_at = pending["expires_at"]
        payload = {
            "action": "email_draft",
            "status": "pending" if pending_id else "success",
            "message": (
                "Immutable email proposal created; explicit approval is required before sending."
                if pending_id
                else "Email draft created. Please preview and confirm before sending."
            ),
            "draft": draft_payload,
            "requires_confirmation": True,
            "require_confirmation": True,
            "pending_id": pending_id,
            "proposal_hash": proposal_hash,
            "expires_at": expires_at,
        }
        return {
            "action": "email_draft",
            "status": payload["status"],
            "message": payload["message"],
            "draft": payload["draft"],
            "require_confirmation": True,
            "response_payload": payload,
            "memory_text": payload["message"],
        }
    except Exception as exc:
        message_text = f"Email draft creation failed: {str(exc)}"
        return {
            "action": "email_draft",
            "status": "error",
            "message": message_text,
            "response_payload": {
                "action": "email_draft",
                "status": "error",
                "message": message_text,
                "msg": message_text,
            },
            "memory_text": message_text,
        }


def _extract_target_strain(message: str, context_snapshot: Any):
    normalized_message = normalize_text_value(message)
    match = re.search(r"\b([A-Za-z][A-Za-z0-9_\-]*[-_][0-9]+)\b", normalized_message)
    if match:
        return match.group(1)
    for strain in getattr(context_snapshot, "strains", []) or []:
        if strain.get("strain_id") and strain.get("strain_id") in normalized_message:
            return strain.get("strain_id")
        if strain.get("name_cn") and strain.get("name_cn") in normalized_message:
            return strain.get("strain_id")
    return None
