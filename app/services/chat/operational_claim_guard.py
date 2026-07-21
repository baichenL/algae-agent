from __future__ import annotations

import re

from app.services.memory.memory_service import append_decision_event


UNVERIFIED_OPERATIONAL_CLAIMS = re.compile(
    r"工具调用已触发|"
    r"正在调用.{0,20}(?:工具|硬件流水线)|"
    r"已(?:成功)?(?:调用工具|创建.{0,12}(?:请求|pending)|执行传代|发送邮件)|"
    r"硬件.{0,12}执行成功",
    re.I | re.S,
)
SAFE_CHAT_REPLACEMENT = "当前只是对话回复，没有检测到真实 Pending 或工具执行回执。"


def guard_chat_reply(
    text: str,
    *,
    session_id: str,
    agent_run_id: str | None,
) -> tuple[str, bool]:
    """Block operational success claims from the provenance-free Chat path."""

    if not UNVERIFIED_OPERATIONAL_CLAIMS.search(text or ""):
        return text, False
    append_decision_event({
        "event": "unsupported_operational_claim_blocked",
        "session_id": session_id,
        "agent_run_id": agent_run_id,
        "route_kind": "chat",
    })
    return SAFE_CHAT_REPLACEMENT, True


def enforce_structured_receipt(response: dict) -> dict:
    """Fail closed when an operational response lacks required provenance."""

    agent_output = response.get("agent_output") or {}
    action = agent_output.get("action")
    if action == "workflow_subculture" and not agent_output.get("pending_id"):
        return {
            "agent_output": {
                "action": "workflow_request",
                "status": "error",
                "reason": "missing_pending_id",
            },
            "natural_reply": "Workflow 请求没有返回真实 pending_id，因此不能声明请求已创建。",
        }
    if action in {"force_subculture", "workflow_execution_result"}:
        tool_result = agent_output.get("tool_result") or agent_output.get("workflow_result")
        if agent_output.get("status") == "success" and not tool_result:
            return {
                "agent_output": {
                    "action": "workflow_execution_result",
                    "status": "error",
                    "reason": "missing_tool_result",
                },
                "natural_reply": "没有检测到真实 tool_result，不能声明 Workflow 已执行。",
            }
    return response
