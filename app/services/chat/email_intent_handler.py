from app.schemas.algae import ChatResponse
from app.services.chat.response_builder import _complete_chat_response
from app.tools.email_tool import has_direct_send_intent, maybe_send_email_from_frontend_intent
from app.services.chat.email_request import EmailRequestSpec, parse_email_request

async def _handle_email_intent_with_tool(
    session_id: str,
    conversation_history: list,
    message: str,
    tool_executor,
    agent_run_id: str | None = None,
    request_spec: dict | None = None,
) -> ChatResponse:
    spec = EmailRequestSpec(**request_spec) if request_spec else parse_email_request(message)
    if spec.missing_fields:
        return _complete_chat_response(
            session_id,
            conversation_history,
            {
                "agent_output": {
                    "action": "email_draft",
                    "status": "needs_more_info",
                    "outcome_status": "needs_input",
                    "missing_fields": list(spec.missing_fields),
                    "email_request_spec": spec.to_dict(),
                    "send": False,
                },
                "natural_reply": "请补充要提醒检查的目标品系；我会继续同一邮件任务，只生成草稿和待审批请求。",
            },
        )
    result = await tool_executor(
        "email_draft",
        {
            "message": spec.source_message or message,
            "session_id": session_id,
            "agent_run_id": agent_run_id,
            "source_message": spec.source_message or message,
            "created_by_tool_call_id": (
                f"email-intent:{session_id}:{agent_run_id or 'untracked'}"
            ),
            "request_approval": bool(spec.create_approval or spec.send),
        },
    )
    agent_report = {
        **result["response_payload"],
        "email_request_spec": spec.to_dict(),
        "target": spec.target,
        "recipient": spec.recipient,
        "send": False,
    }
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
    request_spec: dict | None = None,
) -> ChatResponse | None:
    if tool_executor is not None:
        return _handle_email_intent_with_tool(
            session_id,
            conversation_history,
            message,
            tool_executor,
            agent_run_id,
            request_spec,
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

