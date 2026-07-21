# app/services/context/context_builder.py 
# 负责构建Agent的上下文快照，直接从数据库读取状态和待办列表，生成一个只读的上下文对象供Agent决策和提示使用
# 它不负责判断用户意图，不负责调用工具，不负责修改数据库。
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.time_utils import local_time_string
from app.services.memory.memory_service import (
    get_active_pending_actions,
    get_pending_memory,
    get_recent_experiments,
    get_strain_memory,
    get_status_memory,
)
from app.services.learning.selector import select_learning_context
from app.core.db import assistant_conversations, conversation_tasks

# 定义一个只读的上下文快照数据类，包含当前会话ID、品系状态列表和待办审批列表
@dataclass(frozen=True)
class AgentContextSnapshot:
    """Read-only context bundle for future agent loop and prompt construction."""

    session_id: str
    created_at: str
    strains: List[Dict[str, Any]]
    pending_actions: List[Dict[str, Any]]
    recent_experiments: List[Dict[str, Any]] = field(default_factory=list)
    target_strain: Dict[str, Any] | None = None
    target_pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    selected_agent_memories: List[Dict[str, Any]] = field(default_factory=list)
    selected_user_memories: List[Dict[str, Any]] = field(default_factory=list)
    selected_skill_summaries: List[Dict[str, Any]] = field(default_factory=list)
    learning_budget: Dict[str, int] = field(default_factory=dict)
    task_frame: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "strains": self.strains,
            "pending_actions": self.pending_actions,
            "recent_experiments": self.recent_experiments,
            "target_strain": self.target_strain,
            "target_pending_actions": self.target_pending_actions,
            "selected_agent_memories": self.selected_agent_memories,
            "selected_user_memories": self.selected_user_memories,
            "selected_skill_summaries": self.selected_skill_summaries,
            "learning_budget": self.learning_budget,
            "task_frame": self.task_frame,
        }
    # 将上下文快照转换为一个可供LLM使用的文本块，包含会话ID、创建时间、品系数量、待办事项数量和最近实验数量等信息
    #把DB读取的状态和待办列表格式化成一个文本块，供LLM在构建提示词时使用
    def to_prompt_facts(self) -> str:
        lines = [
            "Algae Agent DB-first context snapshot:",
            f"- session_id: {self.session_id}",
            f"- created_at: {self.created_at}",
            f"- strain_count: {len(self.strains)}",
            f"- pending_action_count: {len(self.pending_actions)}",
            f"- recent_experiment_count: {len(self.recent_experiments)}",
        ]

        if self.target_strain:
            strain = self.target_strain
            lines.append("- target_strain:")
            lines.append(
                "  - "
                f"{strain.get('strain_id')}: "
                f"{strain.get('name_cn')} ({strain.get('name_en')}), "
                f"generation={strain.get('generation_number')}, "
                f"days_since_last_subculture={strain.get('days_since_last_subculture')}, "
                f"last_subculture_time={strain.get('last_subculture_time') or 'unknown'}, "
                f"subculture_due={strain.get('subculture_due')}"
            )
            cycle = strain.get("reminder_cycle") or {}
            if cycle:
                lines.append(
                    "  - reminder_cycle: "
                    f"{cycle.get('cycle_key')}, "
                    f"status={cycle.get('status')}, "
                    f"sent_count={cycle.get('total_sent_count')}, "
                    f"silenced_reason={cycle.get('silenced_reason') or 'none'}"
                )

        if self.task_frame:
            lines.append("- active_task:")
            lines.append(
                "  - "
                f"id={self.task_frame.get('id')}, "
                f"type={self.task_frame.get('task_type')}, "
                f"status={self.task_frame.get('status')}, "
                f"missing={self.task_frame.get('missing_slots') or []}, "
                f"version={self.task_frame.get('version')}"
            )

        if self.target_pending_actions:
            lines.append("- target_pending_actions:")
            for item in self.target_pending_actions[:5]:
                payload = item.get("payload") or {}
                data = payload.get("data") or {}
                lines.append(
                    "  - "
                    f"id={item.get('id')}, "
                    f"type={item.get('action_type')}, "
                    f"target={data.get('strain_id') or 'unknown'}, "
                    f"status={item.get('status')}"
                )

        if self.strains:
            lines.append("- strains:")
            for strain in self.strains:
                lines.append(
                    "  - "
                    f"{strain.get('strain_id')}: "
                    f"{strain.get('name_cn')} ({strain.get('name_en')}), "
                    f"generation={strain.get('generation_number')}, "
                    f"days_since_last_subculture={strain.get('days_since_last_subculture')}, "
                    f"last_subculture_time={strain.get('last_subculture_time') or 'unknown'}, "
                    f"subculture_due={strain.get('subculture_due')}"
                )
                cycle = strain.get("reminder_cycle") or {}
                if cycle:
                    lines.append(
                        "    reminder_cycle: "
                        f"{cycle.get('cycle_key')}, "
                        f"status={cycle.get('status')}, "
                        f"sent_count={cycle.get('total_sent_count')}, "
                        f"silenced_reason={cycle.get('silenced_reason') or 'none'}"
                    )

        if self.pending_actions:
            lines.append("- pending_actions:")
            for item in self.pending_actions[:10]:
                payload = item.get("payload") or {}
                data = payload.get("data") or {}
                lines.append(
                    "  - "
                    f"id={item.get('id')}, "
                    f"type={item.get('action_type')}, "
                    f"target={data.get('strain_id') or 'unknown'}, "
                    f"requester={item.get('requester')}"
                )

        if self.recent_experiments:
            lines.append("- recent_experiments:")
            for experiment in self.recent_experiments[:5]:
                lines.append(
                    "  - "
                    f"id={experiment.get('id')}, "
                    f"strain={experiment.get('strain')}, "
                    f"generation={experiment.get('generation')}, "
                    f"status={experiment.get('status')}, "
                    f"created_at={experiment.get('created_at') or 'unknown'}, "
                    f"hardware_log_count={experiment.get('hardware_log_count', 0)}"
                )

        if self.selected_agent_memories or self.selected_user_memories:
            lines.append("- selected_learning_memories:")
            for memory in (self.selected_agent_memories + self.selected_user_memories)[:5]:
                lines.append(
                    "  - "
                    f"scope={memory.get('scope')}, "
                    f"confidence={memory.get('confidence')}, "
                    f"content={memory.get('content')}"
                )

        if self.selected_skill_summaries:
            lines.append("- selected_skill_summaries:")
            for skill in self.selected_skill_summaries[:5]:
                lines.append(
                    "  - "
                    f"{skill.get('name')}: {skill.get('description')} "
                    f"boundary={skill.get('risk_boundary')}"
                )

        return "\n".join(lines)

# 从数据库读取当前状态和待办列表，构建一个只读的上下文快照对象
def build_context_snapshot(session_id: str, strain_id: str | None = None) -> AgentContextSnapshot:
    """Build a read-only context snapshot without calling an LLM or mutating state."""
    learning = select_learning_context(session_id=session_id, message="")
    return AgentContextSnapshot(
        session_id=session_id,
        created_at=local_time_string(),
        strains=get_status_memory(),
        pending_actions=get_pending_memory(),
        recent_experiments=get_recent_experiments(strain_id=strain_id, limit=5),
        target_strain=get_strain_memory(strain_id) if strain_id else None,
        target_pending_actions=get_active_pending_actions(strain_id=strain_id) if strain_id else [],
        selected_agent_memories=learning.selected_agent_memories,
        selected_user_memories=learning.selected_user_memories,
        selected_skill_summaries=learning.selected_skill_summaries,
        learning_budget=learning.budget,
        task_frame=(
            conversation_tasks.get_active_task(session_id)
            if assistant_conversations.get_conversation(session_id)
            else None
        ),
    )
