from app.schemas.algae import ChatResponse
from app.services.chat.pending_form_state import (
    build_pending_form_state,
    clear_pending_form_state,
    get_pending_form_state,
    merge_pending_form_fields,
    operation_label,
    save_pending_form_state,
)
from app.services.chat.response_builder import _complete_chat_response, _tool_completion_message
from app.services.intent.routing_models import PendingFormDecision, WriteDecision
from app.services.memory.memory_service import append_decision_event
from app.tools.executor import execute_registered_tool


def _find_existing_pending(context_snapshot, action_type: str, strain_id: str) -> dict | None:
    for item in getattr(context_snapshot, "pending_actions", []) or []:
        payload = item.get("payload") or {}
        data = payload.get("data") or {}
        if payload.get("type") == action_type and data.get("strain_id") == strain_id:
            return item
    return None


def _existing_pending_response(existing: dict, args: dict, tool_name: str) -> dict:
    pending_id = existing.get("id")
    action_type = (existing.get("payload") or {}).get("type") or "add_strain"
    action = {
        "add_strain": "add_strain",
        "update_strain": "update_strain",
        "delete_strain": "delete_strain",
    }.get(action_type, action_type)
    prompt = {
        "add_strain": f"Confirm adding this strain?\nAdd strain {args.get('strain_id')}: {args.get('name_cn')} ({args.get('name_en')})",
        "update_strain": f"Confirm applying this update?\nUpdate strain {args.get('strain_id')}",
        "delete_strain": f"Confirm deleting strain {args.get('strain_id')}? This will permanently remove the record after approval.",
    }.get(action_type, "Confirm this pending request?")
    return {
        "agent_output": {
            "action": action,
            "status": "pending",
            "message": f"Pending request already exists, pending_id={pending_id}",
            "msg": f"Pending request already exists, pending_id={pending_id}",
            "pending_id": pending_id,
            "require_confirmation": True,
            "confirmation_prompt": prompt,
            "confirm_endpoint": "/api/v1/strain/confirm",
            "confirm_payload_example": {"pending_id": pending_id, "approve": True},
            "tool_name": tool_name,
        },
        "natural_reply": f"已存在待确认请求 pending {pending_id}，请在前端审批后执行。",
    }


def _pending_more_info_response(state: dict) -> dict:
    missing_fields = state.get("missing_fields") or []
    operation = state.get("operation")
    questions = []
    for field in missing_fields:
        if field == "strain_id":
            questions.append("请提供唯一的 strain_id，或使用能唯一匹配数据库记录的品系名称。")
        elif field == "name_cn":
            questions.append("请提供中文名，例如：螺旋藻。")
        elif field == "name_en":
            questions.append("请提供英文名/拉丁名，例如：Spirulina platensis。")
        elif field == "update_fields":
            questions.append("请说明要修改的字段，例如：品系改为 Chlamydomonas_137AH、代数改为 3，或传代天数改为 5。")
        else:
            questions.append(f"请提供 {field}。")
    message = "；".join(questions) or "请补充必要信息。"
    return {
        "agent_output": {
            "action": "require_more_info",
            "status": "needs_more_info",
            "operation": operation,
            "tool_name": state.get("tool_name"),
            "missing_fields": missing_fields,
            "questions": questions,
            "require_more_info": True,
            "message": message,
            "msg": message,
        },
        "natural_reply": f"{operation_label(operation)}需要补充信息：{message}",
    }


def _pending_conflict_response(state: dict) -> dict:
    label = operation_label(state.get("operation"))
    message = f"当前还有一个未完成的{label}补参流程。请先补充缺失信息，或输入“取消”结束当前流程后再发起新的操作。"
    return {
        "agent_output": {
            "action": "pending_form_conflict",
            "status": "blocked",
            "operation": state.get("operation"),
            "missing_fields": state.get("missing_fields") or [],
            "message": message,
            "msg": message,
        },
        "natural_reply": message,
    }


def _pending_cancel_response() -> dict:
    message = "已取消当前未完成的补参流程。"
    return {
        "agent_output": {
            "action": "pending_form_cancelled",
            "status": "cancelled",
            "message": message,
            "msg": message,
        },
        "natural_reply": message,
    }


def _delete_candidates_response(state: dict) -> dict:
    candidates = state.get("candidates") or []
    lines = [
        "找到多个可能的品系，请确认要删除哪一个：",
        *[f"{index}. {item.get('strain_id')}" for index, item in enumerate(candidates, start=1)],
    ]
    message = "\n".join(lines)
    return {
        "agent_output": {
            "action": "require_more_info",
            "status": "needs_more_info",
            "operation": "delete",
            "tool_name": state.get("tool_name"),
            "missing_fields": ["strain_id"],
            "candidates": candidates,
            "questions": lines[1:],
            "require_more_info": True,
            "message": message,
            "msg": message,
        },
        "natural_reply": message,
    }


async def _execute_pending_tool(
    session_id: str,
    conversation_history: list,
    tool_name: str,
    function_args: dict,
    event_name: str,
    operation: str | None,
    tool_executor=execute_registered_tool,
    agent_run_id: str | None = None,
) -> ChatResponse:
    function_args = {
        **(function_args or {}),
        "session_id": session_id,
        "agent_run_id": agent_run_id,
    }
    execution_result = await tool_executor(tool_name, function_args)
    agent_report = execution_result["response_payload"]
    memory_text = execution_result.get("memory_text", "工具执行完毕")
    completion_message = _tool_completion_message(tool_name, agent_report, memory_text)
    append_decision_event({
        "event": event_name,
        "session_id": session_id,
        "agent_run_id": agent_run_id,
        "route_kind": "write_action",
        "tool": tool_name,
        "operation": operation,
        "strain_id": function_args.get("strain_id"),
        "pending_id": agent_report.get("pending_id"),
    })
    return _complete_chat_response(
        session_id,
        conversation_history,
        {"agent_output": agent_report, "natural_reply": completion_message},
    )


async def handle_pending_form_state(
    session_id: str,
    conversation_history: list,
    message: str,
    decision: PendingFormDecision,
    context_snapshot,
    tool_executor=execute_registered_tool,
    agent_run_id: str | None = None,
) -> ChatResponse:
    active_state = get_pending_form_state(session_id)
    if decision.form_action == "cancel":
        clear_pending_form_state(session_id)
        return _complete_chat_response(session_id, conversation_history, _pending_cancel_response())
    if decision.form_action == "conflict":
        return _complete_chat_response(
            session_id,
            conversation_history,
            _pending_conflict_response(active_state or {}),
        )
    if active_state:
        next_state = merge_pending_form_fields(active_state, message, context_snapshot)
        if next_state.get("operation") == "delete" and not next_state.get("collected_fields", {}).get("strain_id"):
            save_pending_form_state(session_id, next_state)
            if len(next_state.get("candidates") or []) > 1:
                return _complete_chat_response(session_id, conversation_history, _delete_candidates_response(next_state))
            return _complete_chat_response(session_id, conversation_history, _pending_more_info_response(next_state))
        if next_state.get("missing_fields"):
            save_pending_form_state(session_id, next_state)
            return _complete_chat_response(session_id, conversation_history, _pending_more_info_response(next_state))
        function_args = {
            key: value
            for key, value in (next_state.get("collected_fields") or {}).items()
            if value is not None
        }
        clear_pending_form_state(session_id)
        return await _execute_pending_tool(
            session_id,
            conversation_history,
            next_state.get("tool_name"),
            function_args,
            "pending_form_tool_executed",
            next_state.get("operation"),
            tool_executor,
            agent_run_id,
        )

    state = build_pending_form_state(decision, message)
    if state.get("operation") == "delete" and len(state.get("candidates") or []) > 1:
        save_pending_form_state(session_id, state)
        return _complete_chat_response(session_id, conversation_history, _delete_candidates_response(state))
    save_pending_form_state(session_id, state)
    return _complete_chat_response(session_id, conversation_history, _pending_more_info_response(state))


async def handle_create_pending_intent(
    session_id: str,
    conversation_history: list,
    message: str,
    decision: WriteDecision,
    context_snapshot,
    tool_executor=execute_registered_tool,
    agent_run_id: str | None = None,
) -> ChatResponse:
    tool_name = decision.tool_name
    function_args = {
        key: value for key, value in decision.arguments.items() if value is not None
    }
    action_type = {
        "add_algae_strain": "add_strain",
        "update_algae_strain": "update_strain",
        "delete_algae_strain": "delete_strain",
    }.get(tool_name)
    existing = _find_existing_pending(
        context_snapshot,
        action_type,
        function_args.get("strain_id"),
    ) if action_type else None
    if existing:
        return _complete_chat_response(
            session_id,
            conversation_history,
            _existing_pending_response(existing, function_args, tool_name),
        )
    return await _execute_pending_tool(
        session_id,
        conversation_history,
        tool_name,
        function_args,
        "deterministic_pending_tool_executed",
        decision.operation,
        tool_executor,
        agent_run_id,
    )
