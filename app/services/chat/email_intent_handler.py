from app.schemas.algae import ChatResponse
from app.services.chat.response_builder import _complete_chat_response
from app.tools.email_tool import maybe_send_email_from_frontend_intent

async def _handle_email_intent_with_tool(
    session_id: str,
    conversation_history: list,
    message: str,
    tool_executor,
    agent_run_id: str | None = None,
) -> ChatResponse:
    result = await tool_executor(
        "email_draft",
        {
            "message": message,
            "session_id": session_id,
            "agent_run_id": agent_run_id,
            "source_message": message,
        },
    )
    agent_report = result["response_payload"]
    natural_reply = (
        agent_report.get("message")
        or result.get("memory_text")
        or "已生成邮件草稿，请预览后确认发送。"
    )
    if agent_report.get("action") == "email_draft" and agent_report.get("status") == "success":
        natural_reply = "已生成邮件草稿，请预览后确认发送。"
    return _complete_chat_response(
        session_id,
        conversation_history,
        {
            "agent_output": agent_report,
            "natural_reply": natural_reply,
        },
    )


def handle_email_intent(
    session_id: str,
    conversation_history: list,
    message: str,
    context_snapshot,
    tool_executor=None,
    agent_run_id: str | None = None,
) -> ChatResponse | None:
    if tool_executor is not None:
        return _handle_email_intent_with_tool(
            session_id,
            conversation_history,
            message,
            tool_executor,
            agent_run_id,
        )

    result = maybe_send_email_from_frontend_intent(message, context_snapshot)
    if hasattr(result, "sent"):
        response = {
            "agent_output": {
                "action": "email_sent" if result.sent else "email_send_failed",
                **result.model_dump(),
            },
            "natural_reply": result.message,
        }
        return _complete_chat_response(session_id, conversation_history, response)

    response = {
        "agent_output": {
            "action": "email_draft",
            "draft": result.model_dump(),
            "requires_confirmation": True,
        },
        "natural_reply": "已生成邮件草稿，请预览后确认发送。",
    }
    return _complete_chat_response(session_id, conversation_history, response)

