from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.services.agent_runtime.executor import build_agent_action
from app.services.agent_runtime.state import (
    AgentAction,
    AgentRunState,
    AgentStep,
    AgentTerminalStatus,
)


TERMINAL_STOP_STATUSES = {
    AgentTerminalStatus.WAITING_APPROVAL,
    AgentTerminalStatus.NEEDS_MORE_INFO,
    AgentTerminalStatus.FAILED,
    AgentTerminalStatus.BLOCKED,
}


@dataclass(frozen=True)
class AgentLoopDirective:
    should_continue: bool
    reason: str
    terminal_status: AgentTerminalStatus | None = None
    finalize_composite: bool = False

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "should_continue": self.should_continue,
            "reason": self.reason,
            "terminal_status": self.terminal_status.value if self.terminal_status else None,
            "finalize_composite": self.finalize_composite,
        }


def action_signature(action: AgentAction) -> str:
    return json.dumps(
        {
            "action_type": action.action_type,
            "action_name": action.action_name,
            "action_args": action.action_args,
        },
        sort_keys=True,
        default=str,
    )


def should_continue_loop(state: AgentRunState, step: AgentStep) -> AgentLoopDirective:
    if step.terminal_status in TERMINAL_STOP_STATUSES:
        return AgentLoopDirective(
            should_continue=False,
            reason=f"terminal_status:{step.terminal_status.value}",
            terminal_status=step.terminal_status,
        )

    if not state.loop_plan or state.loop_mode != "read_only_composite":
        return AgentLoopDirective(
            should_continue=False,
            reason="no_loop_plan",
            terminal_status=step.terminal_status,
        )

    if not step.policy_verdict or step.policy_verdict.category != "allow_read_only":
        return AgentLoopDirective(
            should_continue=False,
            reason="policy_not_read_only",
            terminal_status=step.terminal_status,
        )

    if state.loop_plan_index >= len(state.loop_plan):
        return AgentLoopDirective(
            should_continue=False,
            reason="loop_plan_complete",
            terminal_status=AgentTerminalStatus.SUCCEEDED,
            finalize_composite=True,
        )

    if state.step_index >= state.max_steps:
        return AgentLoopDirective(
            should_continue=False,
            reason="max_steps_reached",
            terminal_status=AgentTerminalStatus.MAX_STEPS_REACHED,
        )

    next_action = build_agent_action(state.loop_plan[state.loop_plan_index])
    if action_signature(next_action) in state.executed_action_signatures:
        return AgentLoopDirective(
            should_continue=False,
            reason="duplicate_action",
            terminal_status=step.terminal_status,
            finalize_composite=True,
        )

    return AgentLoopDirective(
        should_continue=True,
        reason="next_read_only_plan_step",
        terminal_status=None,
    )
