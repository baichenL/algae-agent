# app/tools/strain_tool_handlers.py
# 负责处理品系相关的 Agent 工具，包括增删改查和待确认请求列表
# 它接收 Agent 的结构化工具参数，但对写操作不直接改数据库，而是创建 pending 审批单
from typing import Any, Dict

from app.services.strains import strain_service
from app.services.memory.memory_service import get_status_memory
from app.tools.registry_core import register_agent_tool
from app.tools.strain_crud import (
    create_add_strain_request,
    create_delete_strain_request,
    create_update_strain_request,
)
from app.tools.tool_schemas import (
    ADD_ALGAE_STRAIN_SCHEMA,
    DELETE_ALGAE_STRAIN_SCHEMA,
    LIST_ALGAE_STRAINS_SCHEMA,
    LIST_PENDING_ACTIONS_SCHEMA,
    UPDATE_ALGAE_STRAIN_SCHEMA,
)


def _tool_result(
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


def _require_more_info(action: str, missing_fields: list[str], questions: list[str], message: str) -> Dict[str, Any]:
    return _tool_result(
        action=action,
        status="needs_more_info",
        message=message,
        require_more_info=True,
        missing_fields=missing_fields,
        questions=questions,
    )


@register_agent_tool(
    name="add_algae_strain",
    schema=ADD_ALGAE_STRAIN_SCHEMA,
    risk_level="medium",
    effect_kind="propose",
    side_effect="db_pending",
    requires_approval=True,
    allowed_callers=["chat_runtime", "frontend_form"],
    exposed_to_llm=True,
    executor_kind="proposal",
    idempotency_fields=["strain_id"],
    audit_event_type="strain_add_requested",
)
def handle_add_strain_tool(function_args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        required = ["strain_id", "name_cn", "name_en"]
        missing = [field for field in required if not function_args.get(field)]
        if missing:
            return _require_more_info(
                "add_strain",
                missing,
                [f"Please provide {field}." for field in missing],
                "More strain information is required to create the pending add request",
            )

        result = create_add_strain_request(function_args)
        pending_id = result.get("pending_id")
        sid = function_args.get("strain_id")
        summary = (
            f"Add strain {sid}: {function_args.get('name_cn')} ({function_args.get('name_en')}), "
            f"generation={function_args.get('generation_number', 1)}, "
            f"days_since_last_subculture={function_args.get('days_since_last_subculture', 0)}"
        )
        return _tool_result(
            action="add_strain",
            status=result.get("status", "pending"),
            message=result.get("msg") or f"Add strain request created, pending_id={pending_id}",
            pending_id=pending_id,
            require_confirmation=True,
            confirmation_prompt=f"Confirm adding this strain?\n{summary}",
            confirm_endpoint="/api/v1/strain/confirm",
            confirm_payload_example={"pending_id": pending_id, "approve": True},
        )
    except Exception as exc:
        return _tool_result("add_strain", "error", f"Add strain failed: {str(exc)}")


@register_agent_tool(
    name="update_algae_strain",
    schema=UPDATE_ALGAE_STRAIN_SCHEMA,
    risk_level="medium",
    effect_kind="propose",
    side_effect="db_pending",
    requires_approval=True,
    allowed_callers=["chat_runtime", "frontend_form"],
    exposed_to_llm=True,
    executor_kind="proposal",
    idempotency_fields=["strain_id"],
    audit_event_type="strain_update_requested",
)
def handle_update_strain_tool(function_args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sid = function_args.get("strain_id")
        if not sid:
            return _require_more_info(
                "update_strain",
                ["strain_id"],
                ["Please provide the strain_id to update"],
                "Missing strain_id",
            )

        updatable = ["new_strain_id", "name_cn", "name_en", "generation_number", "days_since_last_subculture"]
        provided = {key: function_args.get(key) for key in updatable if function_args.get(key) is not None}
        if not provided:
            return _require_more_info(
                "update_strain",
                updatable,
                ["Please provide fields to update, such as name_cn or name_en"],
                "No update fields were provided",
            )

        result = create_update_strain_request(function_args)
        pending_id = result.get("pending_id")
        summary = f"Update strain {sid}: fields {list(provided.keys())}"
        return _tool_result(
            action="update_strain",
            status=result.get("status", "pending"),
            message=result.get("msg") or f"Update strain request created, pending_id={pending_id}",
            pending_id=pending_id,
            require_confirmation=True,
            confirmation_prompt=f"Confirm applying this update?\n{summary}",
            confirm_endpoint="/api/v1/strain/confirm",
            confirm_payload_example={"pending_id": pending_id, "approve": True},
        )
    except Exception as exc:
        return _tool_result("update_strain", "error", f"Update strain failed: {str(exc)}")


@register_agent_tool(
    name="delete_algae_strain",
    schema=DELETE_ALGAE_STRAIN_SCHEMA,
    risk_level="high",
    effect_kind="propose",
    side_effect="db_pending",
    requires_approval=True,
    allowed_callers=["chat_runtime", "frontend_form"],
    exposed_to_llm=True,
    executor_kind="proposal",
    idempotency_fields=["strain_id"],
    audit_event_type="strain_delete_requested",
)
def handle_delete_strain_tool(function_args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sid = function_args.get("strain_id")
        if not sid:
            return _require_more_info(
                "delete_strain",
                ["strain_id"],
                ["Please provide the strain_id to delete"],
                "Missing strain_id",
            )

        result = create_delete_strain_request(function_args)
        pending_id = result.get("pending_id")
        confirm_msg = f"Confirm deleting strain {sid}? This will permanently remove the record after approval."
        return _tool_result(
            action="delete_strain",
            status=result.get("status", "pending"),
            message=result.get("msg") or f"Delete strain request created, pending_id={pending_id}",
            pending_id=pending_id,
            require_confirmation=True,
            confirmation_prompt=confirm_msg,
            confirm_endpoint="/api/v1/strain/confirm",
            confirm_payload_example={"pending_id": pending_id, "approve": True},
        )
    except Exception as exc:
        return _tool_result("delete_strain", "error", f"Delete strain failed: {str(exc)}")


@register_agent_tool(
    name="list_algae_strains",
    schema=LIST_ALGAE_STRAINS_SCHEMA,
    risk_level="none",
    effect_kind="read",
    side_effect="none",
    requires_approval=False,
    allowed_callers=["chat_runtime", "frontend_form", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="domain_fact",
    audit_event_type="strain_list_read",
)
def handle_list_strains_tool(function_args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        items = get_status_memory()
        return _tool_result(
            action="list_strains",
            status="success",
            message=f"Returned {len(items)} strains",
            strains=items,
        )
    except Exception as exc:
        return _tool_result("list_strains", "error", f"List strains failed: {str(exc)}", strains=[])


@register_agent_tool(
    name="list_pending_actions",
    schema=LIST_PENDING_ACTIONS_SCHEMA,
    risk_level="low",
    effect_kind="read",
    side_effect="none",
    requires_approval=False,
    allowed_callers=["chat_runtime", "frontend_form", "mcp_local"],
    exposed_to_llm=True,
    parallel_safe=True,
    executor_kind="read",
    result_authority="control_state",
    audit_event_type="pending_list_read",
)
def handle_list_pending_tool(function_args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        items = strain_service.list_pending_actions()
        return _tool_result(
            action="list_pending",
            status="success",
            message=f"Returned {len(items)} pending requests",
            pending=items,
        )
    except Exception as exc:
        return _tool_result("list_pending", "error", f"List pending requests failed: {str(exc)}", pending=[])
