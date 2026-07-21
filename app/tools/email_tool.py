
import os
import re
from typing import Any, Dict

from app.models.email_schema import EmailSendRequest
from app.services.context.context_builder import build_context_snapshot
from app.services.email.email_service import get_default_recipients, send_email
from app.services.email.email_template import build_manual_check_draft
from app.services.intent.input_normalizer import normalize_text_value
from app.tools.registry_core import register_agent_tool
from app.tools.tool_schemas import EMAIL_DRAFT_SCHEMA


DIRECT_SEND_KEYWORDS = ["\u7acb\u5373\u53d1\u9001", "\u76f4\u63a5\u53d1\u9001", "\u73b0\u5728\u53d1\u9001", "\u9a6c\u4e0a\u53d1\u9001"]


def create_email_draft_from_user_message(message: str, context_snapshot: Any):
    recipients = get_default_recipients()
    metadata = {"strain_id": _extract_target_strain(message, context_snapshot)}
    return build_manual_check_draft(message, recipients, metadata)


def direct_send_enabled() -> bool:
    return os.getenv("EMAIL_FRONTEND_DIRECT_SEND", "false").strip().lower() in {"1", "true", "yes", "on"}


def has_direct_send_intent(message: str) -> bool:
    text = normalize_text_value(message).casefold()
    return any(normalize_text_value(keyword).casefold() in text for keyword in DIRECT_SEND_KEYWORDS)


def maybe_send_email_from_frontend_intent(message: str, context_snapshot: Any):
    draft = create_email_draft_from_user_message(message, context_snapshot)
    if direct_send_enabled() and has_direct_send_intent(message):
        return send_email(EmailSendRequest(
            subject=draft.subject,
            body=draft.body,
            recipients=draft.recipients,
            source="chat_direct_send",
            metadata={**draft.metadata, "template_type": draft.template_type},
        ))
    return draft


@register_agent_tool(
    name="email_draft",
    schema=EMAIL_DRAFT_SCHEMA,
    risk_level="medium",
    effect_kind="draft",
    side_effect="email_draft",
    requires_approval=True,
    allowed_callers=["chat_runtime", "frontend_form"],
    audit_event_type="email_draft_created",
)
def handle_email_draft_tool(function_args: Dict[str, Any]) -> Dict[str, Any]:
    message = function_args.get("message", "")
    session_id = function_args.get("session_id", "default_session")
    context_snapshot = build_context_snapshot(session_id)

    try:
        result = maybe_send_email_from_frontend_intent(message, context_snapshot)
        if hasattr(result, "sent"):
            action = "email_sent" if result.sent else "email_send_failed"
            payload = {
                "action": action,
                "status": result.status,
                "message": result.message,
                **result.model_dump(),
            }
            return {
                "action": action,
                "status": result.status,
                "message": result.message,
                "response_payload": payload,
                "memory_text": result.message,
            }

        draft = result
        payload = {
            "action": "email_draft",
            "status": "success",
            "message": "Email draft created. Please preview and confirm before sending.",
            "draft": draft.model_dump(),
            "requires_confirmation": True,
            "require_confirmation": True,
        }
        return {
            "action": "email_draft",
            "status": "success",
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
