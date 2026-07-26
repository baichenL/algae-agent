from __future__ import annotations

import uuid

from app.core.db import conversation_tasks
from app.core.db.agent_events import record_error_event
from app.schemas.algae import ChatRequest, ChatResponse
from app.services.agent_runtime import AgentRuntimeDeps, run_agent_loop
from app.services.agent_runtime.events import finish_run, record_run_event, start_run
from app.services.agent_runtime.state import RuntimeRequestContext
from app.services.chat.response_builder import _complete_chat_response
from app.services.chat.answer_envelope import attach_answer_envelope
from app.services.chat.task_coordinator import (
    default_request_context,
    resolve_conversation_task,
    synchronize_task_from_response,
)
from app.services.chat.task_spec import build_task_spec, is_email_approval_followup
from app.services.chat.llm_chat_handler import handle_llm_or_tool_path as _handle_llm_or_tool_path
from app.services.chat.pending_form_state import get_pending_form_state
from app.services.chat.workflow_request_state import get_workflow_request_state
from app.services.context.context_builder import build_context_snapshot
from app.services.intent.dispatcher import dispatch_routing_decision
from app.services.intent.input_normalizer import normalize_input
from app.services.intent.intent_handlers import handle_query_status_intent
from app.services.intent.intent_router import build_routing_decision
from app.services.intent.routing_models import RoutingInput
from app.services.intent.routing_models import EmailDecision, ReasonCode, RiskLevel
from app.services.intent.capability_preflight import classify_capability_request
from app.services.chat.email_request import resume_email_request
from app.services.memory.memory_service import append_decision_event, get_session_memory, save_session_memory
from app.tools.executor import execute_registered_tool
from app.tools.email_tool import create_email_pending_from_draft
from app.services.agent_runtime.checkpoint import set_runtime_deps_factory

# 作用是构建运行时依赖，把各种组件装配在一起，形成一个完整的运行时环境
def _build_runtime_deps() -> AgentRuntimeDeps:
    return AgentRuntimeDeps(
        get_session_memory=get_session_memory,
        save_session_memory=save_session_memory,
        build_context_snapshot=build_context_snapshot,
        build_routing_decision=build_routing_decision,
        dispatch_routing_decision=dispatch_routing_decision,
        normalize_input=normalize_input,
        get_pending_form_state=get_pending_form_state,
        get_workflow_request_state=get_workflow_request_state,
        tool_executor=execute_registered_tool,
        query_status_handler=handle_query_status_intent,
        llm_handler=_handle_llm_or_tool_path,
        append_decision_event=append_decision_event,
    )


set_runtime_deps_factory(_build_runtime_deps)


async def handle_chat(
    payload: ChatRequest,
    request_context: RuntimeRequestContext | None = None,
) -> ChatResponse:
    agent_run_id = str(uuid.uuid4())
    effective_context = request_context or default_request_context(payload.session_id)
    active_task = conversation_tasks.get_active_task(
        payload.session_id,
        owner=effective_context.owner,
        workspace_id=effective_context.workspace_id,
    )
    capability_request = classify_capability_request(payload.message)
    if capability_request.forbidden:
        start_run(
            payload.session_id,
            payload.message,
            agent_run_id,
            conversation_id=payload.session_id,
        )
        record_run_event(
            agent_run_id,
            session_id=payload.session_id,
            event_type="run_started",
            layer="agent_runtime",
            payload={"step_index": 1, "stage": "capability_preflight"},
        )
        decision_payload = {
            "route_kind": "policy_rejected",
            "action_type": "policy",
            "action_name": "policy_rejected",
            "risk_level": "high",
            "requires_approval": False,
            "can_continue": False,
            "reason": "requested_capability_outside_safety_envelope",
            "capability_request": capability_request.to_dict(),
        }
        record_run_event(
            agent_run_id,
            session_id=payload.session_id,
            event_type="agent_decision_made",
            layer="agent_runtime",
            payload={"step_index": 1, "decision": decision_payload},
        )
        response = _complete_chat_response(
            payload.session_id,
            get_session_memory(payload.session_id),
            {
                "agent_output": {
                    "action": "policy_rejected",
                    "status": "forbidden",
                    "outcome_status": "failed",
                    "route_kind": "policy_rejected",
                    "reason_code": "policy_capability_forbidden",
                    "capability_request": capability_request.to_dict(),
                    "agent_run_id": agent_run_id,
                },
                "natural_reply": (
                    "我不能覆盖系统或开发者策略、执行原始数据库修改、自行审批，"
                    "也不能进行未经授权的外部执行。这些能力不在当前 Safety Envelope 内，"
                    "且没有创建任务、审批请求或任何领域副作用。"
                ),
            },
        )
        attach_answer_envelope(response)
        finish_run(
            agent_run_id,
            status="blocked",
            final_route="policy_rejected",
            risk_level="high",
            response_summary=response.natural_reply[:500],
        )
        record_run_event(
            agent_run_id,
            session_id=payload.session_id,
            event_type="run_finished",
            layer="chat_service",
            payload={"status": "blocked", "step_index": 1},
        )
        return response
    try:
        context_snapshot = build_context_snapshot(payload.session_id)
        if (
            active_task
            and active_task.get("task_type") == "email"
            and active_task.get("status") == "collecting"
        ):
            email_spec = resume_email_request(active_task, payload.message)
            routing_decision = EmailDecision(
                reason_code=ReasonCode.EMAIL_MATCHED,
                risk_level=RiskLevel.MEDIUM,
                explanation="The message supplies fields for the active email task.",
                source_text=email_spec.source_message,
                request_spec=email_spec.to_dict(),
            )
        else:
            routing_decision = build_routing_decision(
                RoutingInput(
                    message=normalize_input(payload.message),
                    session_id=payload.session_id,
                    context_snapshot=context_snapshot,
                    active_pending_form=get_pending_form_state(payload.session_id),
                    active_workflow_request=get_workflow_request_state(payload.session_id),
                )
            )
    except Exception as exc:
        start_run(payload.session_id, payload.message, agent_run_id)
        record_run_event(
            agent_run_id,
            session_id=payload.session_id,
            event_type="run_started",
            layer="agent_runtime",
            payload={"step_index": 0, "stage": "build_task_spec"},
        )
        error_event_id = record_error_event(
            session_id=payload.session_id,
            agent_run_id=agent_run_id,
            layer="chat_service",
            component="task_spec_preflight",
            operation="build_task_spec",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        record_run_event(
            agent_run_id,
            session_id=payload.session_id,
            event_type="run_failed",
            layer="chat_service",
            payload={
                "step_index": 0,
                "failure_stage": "build_task_spec",
                "failure_layer": "chat_service",
                "error_event_id": error_event_id,
            },
        )
        finish_run(agent_run_id, status="failed", error_event_id=error_event_id)
        raise
    task_spec = build_task_spec(
        message=payload.message,
        decision=routing_decision,
        active_task=active_task,
    )
    resolved = resolve_conversation_task(
        conversation_id=payload.session_id,
        message=payload.message,
        agent_run_id=agent_run_id,
        request_context=effective_context,
        task_spec=task_spec,
    )
    reviewed_draft = (
        ((resolved.task or {}).get("proposed_action") or {}).get("draft")
        if resolved.reused_active
        else None
    )
    if (
        task_spec.task_domain == "email"
        and is_email_approval_followup(payload.message)
        and isinstance(reviewed_draft, dict)
    ):
        pending = create_email_pending_from_draft(
            reviewed_draft,
            agent_run_id=agent_run_id,
            created_by_tool_call_id=f"email-task:{resolved.task['id']}:{agent_run_id}",
            requester=effective_context.owner,
        )
        response = _complete_chat_response(
            payload.session_id,
            get_session_memory(payload.session_id),
            {
                "agent_output": {
                    "action": "email_draft",
                    "status": "pending",
                    "outcome_status": "pending",
                    "draft": reviewed_draft,
                    "pending_id": pending["pending_id"],
                    "proposal_hash": pending["proposal_hash"],
                    "expires_at": pending["expires_at"],
                    "requires_approval": True,
                    "task_spec": task_spec.to_dict(),
                },
                "natural_reply": (
                    f"已基于你审阅的原邮件草稿创建发送审批请求 "
                    f"pending {pending['pending_id']}。邮件尚未发送，需批准后才会执行。"
                ),
            },
        )
        attach_answer_envelope(response)
        task = synchronize_task_from_response(resolved, response)
        response.agent_output["task_id"] = task["id"]
        response.agent_output["task_status"] = task["status"]
        return response
    response = await run_agent_loop(
        payload,
        _build_runtime_deps(),
        # Legacy routes retain their small deterministic loop. Agent Task mode
        # expands this from its server-side Safety Envelope after routing.
        max_steps=3,
        agent_run_id=agent_run_id,
        task_id=resolved.task["id"] if resolved.task else None,
        task_state_version=int(resolved.task["version"]) if resolved.task else None,
        request_context=resolved.request_context,
    )
    attach_answer_envelope(response)
    task = synchronize_task_from_response(resolved, response)
    if task:
        response.agent_output.setdefault("task_id", task["id"])
        response.agent_output.setdefault("task_status", task["status"])
        response.agent_output.setdefault("task_spec", task_spec.to_dict())
        response.agent_output["answer_envelope"]["references"] = [
            *response.agent_output["answer_envelope"].get("references", []),
            {"type": "task", "id": task["id"]},
        ]
        response.agent_output["answer_envelope"]["next_actions"] = [
            {
                **item,
                **(
                    {"task_id": task["id"]}
                    if item.get("action") == "resume_task" and not item.get("task_id")
                    else {}
                ),
            }
            for item in response.agent_output["answer_envelope"].get("next_actions", [])
        ]
    return response
