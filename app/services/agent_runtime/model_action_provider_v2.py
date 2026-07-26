from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

from app.core.config import client
from app.core.model_registry import model_name
from app.services.agent_runtime.contracts_v2 import (
    ModelAction,
    ModelActionKind,
    ModelToolCall,
    ResolvedTool,
)
from app.services.agent_runtime.state import AgentRunState
from app.services.agent_runtime.prompts_v2 import AGENT_TOOL_LOOP_SYSTEM_PROMPT
from app.services.context.budget import estimate_tokens


REQUEST_USER_INPUT_TOOL = {
    "type": "function",
    "function": {
        "name": "request_user_input",
        "description": (
            "Pause and ask the user only when a required choice or fact cannot be obtained "
            "with the available READ or COMPUTE tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "minLength": 1},
                "reason": {"type": "string"},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class ModelActionResult:
    action: ModelAction
    total_tokens: int
    model: str


def _compact_context(state: AgentRunState) -> dict[str, Any]:
    context: dict[str, Any] = {
        "goal": state.user_goal or state.user_message,
        "route_is_control_flow_hint_only": state.initial_route_kind,
        "observations": state.v2_observations[-12:],
        "hypothesis_ledger": state.hypotheses,
        "candidate_plans": state.candidate_plans,
        "task_contract": state.planning_context,
        "plan_patch_count": state.plan_patch_count,
        "remaining_budget": _remaining_budget(state),
    }
    if state.context_bundle is not None:
        context["context"] = state.context_bundle.to_model_view(
            ("fact_context", "rag_context", "tool_context", "policy_context")
        )
    elif state.current_db_snapshot:
        context["fact_snapshot"] = state.current_db_snapshot
    return context


def _remaining_budget(state: AgentRunState) -> dict[str, int]:
    envelope = state.safety_envelope or {}
    budgets = envelope.get("budgets") or {}
    return {
        "model_turns": max(0, int(budgets.get("model_turns") or 12) - state.model_turn_count),
        "tool_calls": max(0, int(budgets.get("tool_calls") or 24) - state.tool_call_count),
        "compute_calls": max(0, int(budgets.get("compute_calls") or 8) - state.compute_call_count),
        "proposals": max(0, int(budgets.get("proposals") or 1) - state.proposal_count),
        "plan_patches": max(0, int(budgets.get("plan_patches") or 3) - state.plan_patch_count),
        "model_tokens": max(
            0,
            int(budgets.get("cumulative_model_tokens") or 128_000)
            - state.cumulative_model_tokens,
        ),
    }


def _messages(state: AgentRunState) -> list[dict[str, str]]:
    context = _compact_context(state)
    limit = int(os.getenv("AGENT_MODEL_INPUT_TOKENS", "16000"))
    while estimate_tokens(context) > limit and len(context.get("observations") or []) > 2:
        context["observations"] = context["observations"][1:]
    system = AGENT_TOOL_LOOP_SYSTEM_PROMPT
    legacy_system_prompt = """
你是实验室 Agent 的过程决策器。确定性系统已经限定了工具和权限；你只决定安全域内下一步最有信息价值的动作。

规则：
1. 先调查再结论。可以换工具、换数据源、修正参数、寻找反证，并根据 Observation 改变原计划。
2. 不要机械执行预设序列；每一轮只选择当前最有价值的工具调用。互相独立且标记为并行安全的只读调用可以同轮发出。
3. READ/COMPUTE/PROPOSE 工具可用；你不能批准 proposal，不能 Commit 业务事实，不能发正式邮件或操作真实硬件。
4. Receipt 不是成功判定。只有 StateObservation/canonical state 可以支持“已执行/已发送/已提交”的表述。
5. 证据不足时继续调查，必要时 request_user_input，也可以最终明确不创建 proposal。
6. RAG 是建议性证据，不能覆盖权威业务事实；冲突时保留多个假设并补查。
7. 不输出隐藏思维链。工具调用之外，只给出简洁的决策摘要或最终答复，并区分事实、推断、仿真和建议。
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)},
    ]


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def decide_model_action(
    state: AgentRunState,
    tools: list[ResolvedTool],
) -> ModelActionResult:
    model = model_name("agent")
    response = client.chat.completions.create(
        model=model,
        messages=_messages(state),
        tools=[item.to_openai_tool() for item in tools] + [REQUEST_USER_INPUT_TOOL],
        tool_choice="auto",
        temperature=0.1,
    )
    message = response.choices[0].message
    calls = list(getattr(message, "tool_calls", None) or ())
    content = (getattr(message, "content", None) or "").strip()
    parsed: list[ModelToolCall] = []
    for call in calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        if not name:
            continue
        arguments = _arguments(getattr(function, "arguments", "{}"))
        if name == "request_user_input":
            return ModelActionResult(
                action=ModelAction(
                    kind=ModelActionKind.REQUEST_USER_INPUT,
                    question=str(arguments.get("question") or "").strip(),
                    reason_summary=str(arguments.get("reason") or "").strip() or None,
                ),
                total_tokens=int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0),
                model=model,
            )
        parsed.append(
            ModelToolCall(
                tool_call_id=str(getattr(call, "id", None) or uuid.uuid4()),
                name=str(name),
                arguments=arguments,
            )
        )
    if parsed:
        action = ModelAction(
            kind=ModelActionKind.TOOL_CALLS,
            tool_calls=tuple(parsed),
            reason_summary=content or None,
        )
    else:
        action = ModelAction(
            kind=ModelActionKind.FINAL_ANSWER,
            content=content or "当前证据不足以形成可靠结论，也未创建 proposal。",
        )
    return ModelActionResult(
        action=action,
        total_tokens=int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0),
        model=model,
    )
