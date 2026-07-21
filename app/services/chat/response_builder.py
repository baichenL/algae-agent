from contextlib import contextmanager
from contextvars import ContextVar

from app.schemas.algae import ChatResponse
from app.services.chat.operational_claim_guard import enforce_structured_receipt
from app.services.memory.memory_service import save_session_memory


PENDING_WRITE_TOOLS = {"add_algae_strain", "update_algae_strain", "delete_algae_strain"}
_DEFER_RESPONSE_PERSISTENCE: ContextVar[bool] = ContextVar(
    "defer_chat_response_persistence",
    default=False,
)


@contextmanager
def defer_response_persistence():
    """Let a composite dispatcher collect child responses and persist once."""

    token = _DEFER_RESPONSE_PERSISTENCE.set(True)
    try:
        yield
    finally:
        _DEFER_RESPONSE_PERSISTENCE.reset(token)


def _complete_chat_response(session_id: str, conversation_history: list, response: dict) -> ChatResponse:
    response = enforce_structured_receipt(response)
    if not _DEFER_RESPONSE_PERSISTENCE.get():
        conversation_history.append({"role": "assistant", "content": response["natural_reply"]})
        save_session_memory(session_id, conversation_history)
    return ChatResponse(
        status="success",
        session_id=session_id,
        agent_output=response["agent_output"],
        natural_reply=response["natural_reply"],
    )


def _tool_completion_message(tool_name: str, agent_report: dict, memory_text: str) -> str:
    status = agent_report.get("status")
    action = agent_report.get("action", tool_name)
    pending_id = agent_report.get("pending_id") or agent_report.get("response_payload", {}).get("pending_id")
    if tool_name in PENDING_WRITE_TOOLS and not pending_id:
        if status == "error":
            return f"请求未创建成功：{agent_report.get('msg') or agent_report.get('message') or memory_text}"
        return "请求未创建成功，后端工具没有返回 pending_id，请检查信息后重试。"
    if status == "pending" and pending_id:
        return f"已创建待确认请求 pending {pending_id}，请在前端审批后执行。"
    if agent_report.get("require_confirmation") and pending_id:
        return f"待确认请求 pending {pending_id} 已准备好，请在前端审批后执行。"
    if agent_report.get("require_confirmation"):
        return "该操作需要确认，但系统没有返回 pending_id，请刷新 pending 列表核对后再操作。"
    if status == "success":
        return "已完成。"
    if status == "error":
        return f"执行失败：{agent_report.get('msg') or memory_text}"
    return memory_text or "工具执行完毕。"

