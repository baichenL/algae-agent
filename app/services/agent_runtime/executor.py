from __future__ import annotations

from app.schemas.algae import ChatResponse
from app.services.chat.response_builder import _complete_chat_response
from app.services.agent_runtime.state import (
    AgentAction,
    AgentDecision,
    AgentObservation,
    AgentPolicyVerdict,
    AgentRunState,
    AgentRuntimeDeps,
    AgentTerminalStatus,
)
from app.services.intent.dispatcher import DispatchContext


def build_agent_action(decision: AgentDecision) -> AgentAction:
    return AgentAction(
        action_type=decision.action_type,
        action_name=decision.action_name,
        action_args=decision.action_args,
        risk_level=decision.risk_level,
        requires_approval=decision.requires_approval,
        raw_decision=decision.raw_decision,
    )


async def execute_agent_action(
    action: AgentAction,
    state: AgentRunState,
    deps: AgentRuntimeDeps,
) -> ChatResponse:
    source_message = getattr(action.raw_decision, "source_text", None) or state.user_message
    return await deps.dispatch_routing_decision(
        action.raw_decision,
        DispatchContext(
            session_id=state.session_id,
            conversation_history=state.conversation_history,
            original_message=source_message,
            context_snapshot=state.context_snapshot,
            agent_run_id=state.agent_run_id,
            tool_executor=deps.tool_executor,
            query_status_handler=deps.query_status_handler,
            llm_handler=deps.llm_handler,
        ),
    )


def build_composite_loop_response(
    state: AgentRunState,
    deps: AgentRuntimeDeps,
) -> ChatResponse:
    step_payloads = [
        {
            "route_kind": observation.route_kind,
            "action": observation.action,
            "status": observation.status,
            "agent_output": observation.output,
            "natural_reply": observation.natural_reply,
        }
        for observation in state.previous_observations
    ]
    natural_reply = "\n\n".join(
        f"{index}. {observation.natural_reply}"
        for index, observation in enumerate(state.previous_observations, start=1)
        if observation.natural_reply
    )
    if not natural_reply:
        natural_reply = "已完成多步只读查询。"
    return _complete_chat_response(
        state.session_id,
        state.conversation_history,
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


def build_blocked_response(
    action: AgentAction,
    verdict: AgentPolicyVerdict,
    state: AgentRunState,
    deps: AgentRuntimeDeps,
) -> ChatResponse:
    if verdict.category == "existing_approval_boundary":
        pending_id = verdict.evidence.get("pending_id")
        target = verdict.evidence.get("target")
        natural_reply = (
            f"An existing workflow approval request already covers {target or 'this target'}"
            f"{f' (pending {pending_id})' if pending_id else ''}. "
            "No duplicate pending request will be created."
        )
        output = {
            "action": action.action_name,
            "status": "pending",
            "pending_id": pending_id,
            "target": target,
            "require_confirmation": True,
            "policy": verdict.to_event_payload(),
        }
        state.conversation_history.append({"role": "assistant", "content": natural_reply})
        deps.save_session_memory(state.session_id, state.conversation_history)
        return ChatResponse(
            status="success",
            session_id=state.session_id,
            agent_output=output,
            natural_reply=natural_reply,
        )

    natural_reply = "该请求已被 Agent Runtime Policy Guard 阻断，不会执行任何工具、工作流或数据库副作用。"
    output = {
        "action": action.action_name,
        "status": "blocked",
        "message": natural_reply,
        "policy": verdict.to_event_payload(),
    }
    state.conversation_history.append({"role": "assistant", "content": natural_reply})
    deps.save_session_memory(state.session_id, state.conversation_history)
    return ChatResponse(
        status="success",
        session_id=state.session_id,
        agent_output=output,
        natural_reply=natural_reply,
    )


def collect_observation(
    response: ChatResponse,
    decision: AgentDecision,
) -> AgentObservation:
    output = response.agent_output or {}
    pending_id = output.get("pending_id")
    try:
        pending_id = int(pending_id) if pending_id is not None else None
    except (TypeError, ValueError):
        pending_id = None
    return AgentObservation(
        status=output.get("status", response.status),
        action=output.get("action"),
        route_kind=decision.route_kind,
        output=output,
        natural_reply=response.natural_reply,
        pending_id=pending_id,
        error_event_id=output.get("error_event_id"),
    )


def derive_terminal_status(observation: AgentObservation) -> AgentTerminalStatus:
    output = observation.output or {}
    status = observation.status
    if status == "pending" or output.get("require_confirmation") or output.get("requires_confirmation"):
        return AgentTerminalStatus.WAITING_APPROVAL
    if status in {"needs_more_info", "needs_clarification"} or output.get("require_more_info"):
        return AgentTerminalStatus.NEEDS_MORE_INFO
    if status == "error":
        return AgentTerminalStatus.FAILED
    if status == "blocked":
        return AgentTerminalStatus.BLOCKED
    return AgentTerminalStatus.SUCCEEDED
