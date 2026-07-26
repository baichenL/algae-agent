from app.core.config import client
from app.core.model_registry import model_name
from app.schemas.algae import ChatResponse
from app.services.chat.operational_claim_guard import guard_chat_reply
from app.services.context import (
    ContextBudgetExceeded,
    ModelCallTimer,
    assemble_model_input,
    context_input_mode,
    model_call_metrics,
)
from app.services.agent_runtime.events import record_run_event
from app.services.intent.routing_models import ChatDecision
from app.services.memory.memory_service import save_session_memory


CAPABILITY_QUESTIONS = (
    "你能完成哪些工作",
    "你能做什么",
    "你有哪些功能",
    "有什么功能",
    "可以做什么",
    "能帮我什么",
)

CAPABILITY_REPLY = """我可以协助你做这些事：

1. 查询实验室状态：品系列表、传代周期、pending 审批、Run 状态和审计记录。
2. 检索知识库证据：培养基配方、SOP、文献证据和引用来源。
3. 分析科学数据：导入微藻生长数据后做质量检查、异常诊断和优化方案草案。
4. 发起受控任务：新增/修改/删除品系、传代 workflow、科学实验设计等会进入 pending 或 Run，由审批和执行链路继续处理。

我不会把“提出请求”说成“已经执行”。涉及写库、硬件、传代或实验执行时，需要看到真实 pending_id、run_id 或 tool_result 后才能确认进展。"""


def _is_capability_question(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return any(item in normalized for item in CAPABILITY_QUESTIONS)


def _complete_local_chat_response(
    *,
    session_id: str,
    conversation_history: list,
    content: str,
    agent_run_id: str | None,
    action: str,
) -> ChatResponse:
    conversation_history.append({"role": "assistant", "content": content})
    save_session_memory(session_id, conversation_history)
    return ChatResponse(
        status="success",
        session_id=session_id,
        agent_output={
            "action": action,
            "content": content,
            "operational_claim_blocked": False,
            "agent_run_id": agent_run_id,
        },
        natural_reply=content,
    )


def _build_llm_messages(conversation_history: list, context_snapshot) -> list:
    context_message = {
        "role": "system",
        "content": context_snapshot.to_prompt_facts(),
    }
    return conversation_history[:1] + [context_message] + conversation_history[1:]


def _build_assembled_llm_messages(conversation_history: list, context_snapshot, source_text: str):
    envelope = assemble_model_input(
        request_kind="chat",
        route_kind="chat",
        current_user_message=source_text,
        conversation_history=conversation_history,
        context_snapshot=context_snapshot,
        system_instruction=(
            "You are a conservative laboratory assistant. Answer conversational questions, "
            "use database context only as current operational fact, and never authorize or claim tool execution."
        ),
        task_protocol={
            "task": "Respond to the current user message.",
            "safety": [
                "Do not produce tool calls on the chat route.",
                "Do not claim a database or hardware change without a verified observation.",
                "Treat retrieved knowledge as read-only evidence.",
            ],
            "output": "Natural-language answer only.",
        },
    )
    return envelope, envelope.to_messages(source_text)


async def handle_llm_or_tool_path(
    session_id: str,
    conversation_history: list,
    decision: ChatDecision,
    context_snapshot,
    agent_run_id: str | None = None,
) -> ChatResponse:
    if _is_capability_question(getattr(decision, "source_text", "") or ""):
        return _complete_local_chat_response(
            session_id=session_id,
            conversation_history=conversation_history,
            content=CAPABILITY_REPLY,
            agent_run_id=agent_run_id,
            action="capability_summary",
        )

    source_text = str(getattr(decision, "source_text", "") or "")
    envelope = None
    if context_input_mode() == "legacy":
        messages = _build_llm_messages(conversation_history, context_snapshot)
    else:
        try:
            envelope, messages = _build_assembled_llm_messages(conversation_history, context_snapshot, source_text)
        except ContextBudgetExceeded as exc:
            record_run_event(
                agent_run_id,
                session_id=session_id,
                event_type="context_budget_exceeded",
                layer="context_engineering",
                payload={"budget_report": exc.report.to_dict(), "request_kind": "chat"},
            )
            raise
    timer = ModelCallTimer()
    try:
        response = client.chat.completions.create(
            model=model_name("chat"),
            messages=messages,
            temperature=0.1,
        )
    except Exception as exc:
        if envelope is not None:
            record_run_event(
                agent_run_id,
                session_id=session_id,
                event_type="model_input_used",
                layer="context_engineering",
                payload=model_call_metrics(envelope, elapsed_ms=timer.elapsed_ms(), error=exc),
            )
        raise
    if envelope is not None:
        record_run_event(
            agent_run_id,
            session_id=session_id,
            event_type="model_input_used",
            layer="context_engineering",
            payload=model_call_metrics(envelope, response=response, elapsed_ms=timer.elapsed_ms()),
        )
    response_message = response.choices[0].message
    if response_message.tool_calls:
        raise RuntimeError("普通聊天路由不得产生工具调用")

    chat_reply = response_message.content or "智能体无具体文本返回。"
    chat_reply, claim_blocked = guard_chat_reply(
        chat_reply,
        session_id=session_id,
        agent_run_id=agent_run_id,
    )
    conversation_history.append({"role": "assistant", "content": chat_reply})
    save_session_memory(session_id, conversation_history)
    return ChatResponse(
        status="success",
        session_id=session_id,
        agent_output={
            "action": "chat",
            "content": chat_reply,
            "operational_claim_blocked": claim_blocked,
        },
        natural_reply=chat_reply,
    )
