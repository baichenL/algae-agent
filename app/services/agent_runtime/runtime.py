from __future__ import annotations

from app.schemas.algae import ChatRequest, ChatResponse
from app.services.agent_runtime.graph_runtime import run_agent_graph_loop
from app.services.agent_runtime.state import AgentRuntimeDeps, RuntimeRequestContext


async def run_agent_loop(
    payload: ChatRequest,
    deps: AgentRuntimeDeps,
    max_steps: int = 8,
    agent_run_id: str | None = None,
    task_id: str | None = None,
    task_state_version: int | None = None,
    request_context: RuntimeRequestContext | None = None,
) -> ChatResponse:
    """Run the chat agent through the LangGraph orchestration layer.

    The public entry point stays stable for chat_service and tests, while the
    internal control flow is now explicit graph nodes: context, decision,
    policy, execution, observation, and replan.
    """

    return await run_agent_graph_loop(
        payload,
        deps,
        max_steps=max_steps,
        agent_run_id=agent_run_id,
        task_id=task_id,
        task_state_version=task_state_version,
        request_context=request_context,
    )
