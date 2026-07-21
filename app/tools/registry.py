
# app/tools/registry.py
# 负责统一管理 Agent 工具的注册入口
"""Unified entrypoint for agent tool registration.

Importing this module loads all real tool handlers so existing call sites can
use AGENT_TOOL_REGISTRY unchanged. Helper functions are intentionally not
registered as tools.
"""

from app.tools.registry_core import AGENT_TOOL_REGISTRY, register_agent_tool
from app.tools.tool_schemas import TRIGGER_SUBCULTURE_WORKFLOW_SCHEMA
from app.tools.workflow_tool_handlers import handle_subculture_tool
from app.tools.strain_tool_handlers import (
    handle_add_strain_tool,
    handle_delete_strain_tool,
    handle_list_pending_tool,
    handle_list_strains_tool,
    handle_update_strain_tool,
)
from app.tools.email_tool import handle_email_draft_tool
from app.tools.scientific_tool_handlers import (
    handle_experiment_design_preview,
    handle_experiment_proposal_request,
    handle_growth_diagnosis_start,
    handle_scientific_datasets_list,
)


if "trigger_subculture_workflow" not in AGENT_TOOL_REGISTRY:
    register_agent_tool(
        name="trigger_subculture_workflow",
        schema=TRIGGER_SUBCULTURE_WORKFLOW_SCHEMA,
        risk_level="high",
        effect_kind="propose",
        side_effect="hardware_workflow",
        requires_approval=True,
        requires_explicit_confirmation=True,
        allowed_callers=["chat_runtime", "workflow_approval_service"],
        idempotency_fields=["strain_id", "agent_run_id"],
        audit_event_type="workflow_subculture_requested",
    )(handle_subculture_tool)

__all__ = [
    "AGENT_TOOL_REGISTRY",
    "register_agent_tool",
    "handle_subculture_tool",
    "handle_add_strain_tool",
    "handle_update_strain_tool",
    "handle_delete_strain_tool",
    "handle_list_strains_tool",
    "handle_list_pending_tool",
    "handle_email_draft_tool",
    "handle_scientific_datasets_list",
    "handle_growth_diagnosis_start",
    "handle_experiment_design_preview",
    "handle_experiment_proposal_request",
]
