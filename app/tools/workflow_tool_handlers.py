# app/tools/workflow_tool_handlers.py
# 负责处理与工作流相关的 Agent 工具，目前包含一个触发传代流程的工具
from typing import Any, Dict

from app.services.protocols.pending_payload import build_protocol_response_fields
from app.services.strains import strain_service
from app.tools.registry_core import register_agent_tool
from app.tools.tool_schemas import TRIGGER_SUBCULTURE_WORKFLOW_SCHEMA


def _workflow_result(
    action: str,
    status: str,
    message: str,
    response_payload: Dict[str, Any] | None = None,
    **extra: Any,
) -> Dict[str, Any]:
    payload = {"action": action, "status": status, "message": message, "msg": message}
    if response_payload:
        payload.update(response_payload)
    payload.update(extra)
    return {
        "action": action,
        "status": status,
        "message": message,
        **extra,
        "response_payload": payload,
        "memory_text": message,
    }


@register_agent_tool(
    name="trigger_subculture_workflow",
    schema=TRIGGER_SUBCULTURE_WORKFLOW_SCHEMA,
    risk_level="high",
    effect_kind="propose",
    side_effect="hardware_workflow",
    requires_approval=True,
    requires_explicit_confirmation=True,
    allowed_callers=["chat_runtime", "workflow_approval_service"],
    idempotency_fields=["strain_id", "agent_run_id"],
    audit_event_type="workflow_subculture_requested",
)
async def handle_subculture_tool(function_args: Dict[str, Any]) -> Dict[str, Any]:
    strain = function_args.get("strain_id")
    if not strain:
        return _workflow_result(
            action="workflow_request",
            status="error",
            message="A verified strain_id is required before creating a workflow pending request.",
        )
    try:
        pending_id = strain_service.create_pending_workflow_subculture(
            strain,
            requester="agent_tool",
            session_id=function_args.get("session_id"),
            agent_run_id=function_args.get("agent_run_id"),
            source_message=function_args.get("source_message"),
            graph_thread_id=function_args.get("graph_thread_id"),
            execution_idempotency_key=function_args.get("execution_idempotency_key"),
            domain_dedupe_key=function_args.get("domain_dedupe_key"),
        )
        pending = strain_service.get_pending_action(pending_id)
        if not pending or pending.get("status") != "pending":
            raise RuntimeError("pending_verification_failed")
        protocol_fields = build_protocol_response_fields((pending or {}).get("payload") or {})
        return _workflow_result(
            action="workflow_subculture",
            status="pending",
            message=(
                f"Created workflow pending request {pending_id}; protocol validation "
                "and simulation preview completed before approval."
            ),
            pending_id=pending_id,
            strain_id=strain,
            require_confirmation=True,
            requires_approval=True,
            **protocol_fields,
        )
    except Exception as exc:
        return _workflow_result(
            action="workflow_request",
            status="error",
            message=f"Workflow pending request failed: {str(exc)}",
            strain_id=strain,
        )
