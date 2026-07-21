from __future__ import annotations

import inspect
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Awaitable, Callable

from app.schemas.algae import ChatResponse
from app.services.chat.email_intent_handler import handle_email_intent
from app.services.chat.pending_intent_handler import handle_create_pending_intent, handle_pending_form_state
from app.services.chat.query_intent_handler import handle_query_intent, handle_tool_info_intent
from app.services.chat.rag_intent_handler import handle_rag_intent
from app.services.chat.response_builder import _complete_chat_response, defer_response_persistence
from app.services.chat.workflow_intent_handler import (
    handle_workflow_audit_intent,
    handle_workflow_intent,
)
from app.services.chat.workflow_request_state import (
    build_workflow_request_state,
    save_workflow_request_state,
)
from app.services.intent.routing_models import (
    ChatDecision,
    ClarificationDecision,
    CompositeDecision,
    EmailDecision,
    KnowledgeDecision,
    PendingFormDecision,
    QueryDecision,
    RoutingDecision,
    RiskLevel,
    RouteKind,
    ReasonCode,
    ToolInfoDecision,
    WorkflowDecision,
    WorkflowAuditDecision,
    WriteDecision,
    ScientificTaskDecision,
)


@dataclass(frozen=True)
class DispatchContext:
    session_id: str
    conversation_history: list
    original_message: str
    context_snapshot: Any
    agent_run_id: str
    tool_executor: Callable[..., Awaitable[dict]]
    query_status_handler: Callable[..., dict]
    llm_handler: Callable[..., Awaitable[ChatResponse]]


def _source_message(decision, ctx: DispatchContext) -> str:
    # The top-level context carries the byte-for-byte user message. Composite
    # dispatch replaces it with the normalized fragment for each child step.
    return ctx.original_message


async def _dispatch_tool_info(decision: ToolInfoDecision, ctx: DispatchContext) -> ChatResponse:
    return handle_tool_info_intent(ctx.session_id, ctx.conversation_history, decision)


async def _dispatch_email(decision: EmailDecision, ctx: DispatchContext) -> ChatResponse:
    try:
        response = handle_email_intent(
            ctx.session_id,
            ctx.conversation_history,
            _source_message(decision, ctx),
            ctx.context_snapshot,
            ctx.tool_executor,
            ctx.agent_run_id,
        )
    except TypeError:
        # Backward compatibility for tests or callers monkeypatching the legacy
        # four-argument handler.
        response = handle_email_intent(
            ctx.session_id,
            ctx.conversation_history,
            _source_message(decision, ctx),
            ctx.context_snapshot,
        )
    if inspect.isawaitable(response):
        response = await response
    if response is None:
        raise RuntimeError("Email decision did not produce an email response")
    return response


async def _dispatch_query(decision: QueryDecision, ctx: DispatchContext) -> ChatResponse:
    return handle_query_intent(
        ctx.session_id,
        ctx.conversation_history,
        decision,
        ctx.context_snapshot,
        ctx.query_status_handler,
    )


async def _dispatch_pending_form(decision: PendingFormDecision, ctx: DispatchContext) -> ChatResponse:
    return await handle_pending_form_state(
        ctx.session_id,
        ctx.conversation_history,
        _source_message(decision, ctx),
        decision,
        ctx.context_snapshot,
        ctx.tool_executor,
        ctx.agent_run_id,
    )


async def _dispatch_knowledge(decision: KnowledgeDecision, ctx: DispatchContext) -> ChatResponse:
    try:
        return handle_rag_intent(
            ctx.session_id,
            ctx.conversation_history,
            _source_message(decision, ctx),
            ctx.agent_run_id,
        )
    except TypeError:
        return handle_rag_intent(
            ctx.session_id,
            ctx.conversation_history,
            _source_message(decision, ctx),
        )


async def _dispatch_write(decision: WriteDecision, ctx: DispatchContext) -> ChatResponse:
    return await handle_create_pending_intent(
        ctx.session_id,
        ctx.conversation_history,
        _source_message(decision, ctx),
        decision,
        ctx.context_snapshot,
        ctx.tool_executor,
        ctx.agent_run_id,
    )


async def _dispatch_workflow(decision: WorkflowDecision, ctx: DispatchContext) -> ChatResponse:
    return await handle_workflow_intent(
        ctx.session_id,
        ctx.conversation_history,
        decision,
        ctx.agent_run_id,
        _source_message(decision, ctx),
        ctx.tool_executor,
    )


async def _dispatch_workflow_audit(
    decision: WorkflowAuditDecision,
    ctx: DispatchContext,
) -> ChatResponse:
    return handle_workflow_audit_intent(
        ctx.session_id,
        ctx.conversation_history,
        decision,
    )


async def _dispatch_scientific(decision: ScientificTaskDecision, ctx: DispatchContext) -> ChatResponse:
    from app.services.scientific.service import run_scientific_task

    if decision.missing_fields or not decision.arguments.get("dataset_id"):
        return _complete_chat_response(
            ctx.session_id,
            ctx.conversation_history,
            {
                "agent_output": {
                    "action": "scientific_dataset_required",
                    "status": "needs_more_info",
                    "missing_fields": ["dataset_id"],
                },
                "natural_reply": "请先在 Scientific Workbench 导入 CSV/XLSX 生长数据，或在请求中提供唯一 dataset_id。",
            },
        )
    result = run_scientific_task(
        dataset_id=decision.arguments["dataset_id"],
        mode=decision.arguments.get("mode") or "diagnose",
        offline_replay=bool(decision.arguments.get("offline_replay")),
        session_id=ctx.session_id,
        agent_run_id=ctx.agent_run_id,
        create_pending=(decision.arguments.get("mode") == "diagnose_and_optimize"),
    )
    proposal = result.get("proposal") or {}
    status = "pending" if proposal.get("pending_id") else result.get("status", "success")
    output = {
        "action": "scientific_closed_loop",
        "status": status,
        "scientific_run_id": result.get("id"),
        "pending_id": proposal.get("pending_id"),
        "require_confirmation": bool(proposal.get("pending_id")),
        "simulation_only": True,
        "artifact_count": len(result.get("artifacts") or []),
    }
    reply = (
        f"科学闭环已运行到人工审批边界：scientific_run={result.get('id')}，pending={proposal.get('pending_id')}。"
        if proposal.get("pending_id")
        else f"科学诊断已完成：scientific_run={result.get('id')}。结论和验证报告已写入可追踪 artifacts。"
    )
    return _complete_chat_response(
        ctx.session_id, ctx.conversation_history,
        {"agent_output": output, "natural_reply": reply},
    )


async def _dispatch_clarification(
    decision: ClarificationDecision,
    ctx: DispatchContext,
) -> ChatResponse:
    if _should_collect_workflow_target(decision):
        state = build_workflow_request_state(
            ctx.session_id,
            ctx.agent_run_id,
            ctx.original_message,
        )
        save_workflow_request_state(ctx.session_id, state)
    if decision.response_action == "ambiguous_intent":
        message = "你提到了传代。请问你是想了解传代相关知识，还是要执行某个明确品系的传代流程？如果要执行，请提供 strain_id。"
    else:
        message = "\n".join(decision.questions) or "请明确本次要处理的唯一查询或操作。"
    return _complete_chat_response(
        ctx.session_id,
        ctx.conversation_history,
        {
            "agent_output": {
                "action": decision.response_action,
                "status": (
                    "success"
                    if decision.response_action in {"dynamic_replan_explain_result", "dynamic_replan_stop_reason"}
                    else "needs_clarification"
                ),
                "reason_code": decision.reason_code.value,
                "risk_level": decision.risk_level.value,
                "candidate_routes": [item.kind.value for item in decision.candidates],
                "questions": list(decision.questions),
            },
            "natural_reply": message,
        },
    )


def _should_collect_workflow_target(decision: ClarificationDecision) -> bool:
    if decision.response_action == "workflow_request_clarification":
        return True
    if decision.risk_level != RiskLevel.HIGH:
        return False
    if decision.reason_code not in {
        ReasonCode.MULTIPLE_TARGETS_CONFLICT,
        ReasonCode.WORKFLOW_TARGET_REQUIRED,
    }:
        return False
    return any(item.kind == RouteKind.WORKFLOW for item in decision.candidates)


async def _dispatch_chat(decision: ChatDecision, ctx: DispatchContext) -> ChatResponse:
    return await ctx.llm_handler(
        ctx.session_id,
        ctx.conversation_history,
        decision,
        ctx.context_snapshot,
        ctx.agent_run_id,
    )


async def _dispatch_composite(
    decision: CompositeDecision,
    ctx: DispatchContext,
) -> ChatResponse:
    results: list[tuple[object, ChatResponse]] = []
    with defer_response_persistence():
        for step in decision.steps:
            handler = _HANDLERS.get(step.kind)
            if handler is None or step.kind == RouteKind.COMPOSITE:
                raise RuntimeError(f"No composite step dispatcher registered for route: {step.kind}")
            step_context = replace(
                ctx,
                original_message=step.source_text or ctx.original_message,
            )
            results.append((step, await handler(step, step_context)))

    step_payloads = [
        {
            "route_kind": step.kind.value,
            "action": response.agent_output.get("action"),
            "status": response.agent_output.get("status", response.status),
            "agent_output": response.agent_output,
            "natural_reply": response.natural_reply,
        }
        for step, response in results
    ]
    natural_reply = "\n\n".join(
        f"{index}. {response.natural_reply}"
        for index, (_, response) in enumerate(results, start=1)
    )
    return _complete_chat_response(
        ctx.session_id,
        ctx.conversation_history,
        {
            "agent_output": {
                "action": "composite_result",
                "status": "success",
                "step_count": len(step_payloads),
                "steps": step_payloads,
            },
            "natural_reply": natural_reply,
        },
    )


_HANDLERS = {
    RouteKind.TOOL_INFO: _dispatch_tool_info,
    RouteKind.EMAIL: _dispatch_email,
    RouteKind.LAB_QUERY: _dispatch_query,
    RouteKind.PENDING_QUERY: _dispatch_query,
    RouteKind.PENDING_FORM: _dispatch_pending_form,
    RouteKind.KNOWLEDGE_QUERY: _dispatch_knowledge,
    RouteKind.WRITE_ACTION: _dispatch_write,
    RouteKind.WORKFLOW: _dispatch_workflow,
    RouteKind.WORKFLOW_AUDIT: _dispatch_workflow_audit,
    RouteKind.SCIENTIFIC_TASK: _dispatch_scientific,
    RouteKind.CLARIFICATION: _dispatch_clarification,
    RouteKind.CHAT: _dispatch_chat,
    RouteKind.COMPOSITE: _dispatch_composite,
}


async def dispatch_routing_decision(
    decision: RoutingDecision,
    context: DispatchContext,
) -> ChatResponse:
    handler = _HANDLERS.get(decision.kind)
    if handler is None:
        raise RuntimeError(f"No dispatcher registered for route: {decision.kind}")
    return await handler(decision, context)
