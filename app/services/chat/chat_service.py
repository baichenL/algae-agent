from __future__ import annotations

import uuid

from app.schemas.algae import ChatRequest, ChatResponse
from app.services.agent_runtime import AgentRuntimeDeps, run_agent_loop
from app.services.agent_runtime.state import RuntimeRequestContext
from app.services.chat.task_coordinator import resolve_conversation_task, synchronize_task_from_response
from app.services.chat.llm_chat_handler import handle_llm_or_tool_path as _handle_llm_or_tool_path
from app.services.chat.pending_form_state import get_pending_form_state
from app.services.chat.workflow_request_state import get_workflow_request_state
from app.services.context.context_builder import build_context_snapshot
from app.services.intent.dispatcher import dispatch_routing_decision
from app.services.intent.input_normalizer import normalize_input
from app.services.intent.intent_handlers import handle_query_status_intent
from app.services.intent.intent_router import build_routing_decision
from app.services.memory.memory_service import append_decision_event, get_session_memory, save_session_memory
from app.tools.executor import execute_registered_tool
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
    resolved = resolve_conversation_task(
        conversation_id=payload.session_id,
        message=payload.message,
        agent_run_id=agent_run_id,
        request_context=request_context,
    )
    response = await run_agent_loop(
        payload,
        _build_runtime_deps(),
        max_steps=3,
        agent_run_id=agent_run_id,
        task_id=resolved.task["id"] if resolved.task else None,
        task_state_version=int(resolved.task["version"]) if resolved.task else None,
        request_context=resolved.request_context,
    )
    task = synchronize_task_from_response(resolved, response)
    if task:
        response.agent_output.setdefault("task_id", task["id"])
        response.agent_output.setdefault("task_status", task["status"])
    return response
