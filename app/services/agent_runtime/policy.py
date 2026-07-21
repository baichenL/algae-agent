from __future__ import annotations

from app.services.agent_runtime.state import AgentAction, AgentPolicyVerdict, AgentRunState
from app.services.intent.routing_models import RouteKind
from app.tools.registry import AGENT_TOOL_REGISTRY
from app.services.agent_runtime.toolsets import context_allows_tool


READ_ONLY_ACTIONS = {
    "list_algae_strains",
    "list_pending_actions",
    "retrieve_rag",
    "read_experiment_logs",
    "read_workflow_runs",
    "tool_info",
    "lab_query",
    "pending_query",
    "knowledge_query",
    "workflow_audit",
    "generation_number",
    "list_strains",
    "pending",
    "reminder_status",
    "strain_status",
    "due_subculture",
}

REQUIRES_APPROVAL_ACTIONS = {
    "add_algae_strain",
    "update_algae_strain",
    "delete_algae_strain",
    "email",
    "send_email",
    "email_draft",
    "trigger_subculture_workflow",
    "workflow_subculture",
    "workflow",
}

FORBIDDEN_DIRECT_ACTIONS = {
    "commit_db_change",
    "trigger_hardware",
    "delete_real_data",
    "approve_pending_from_chat",
    "commit_workflow_result_from_chat",
}

INTERNAL_SAFE_ACTION_TYPES = {
    "chat",
    "clarification",
    "pending_form",
    "composite",
}

KNOWN_ACTION_TYPES = {
    "read",
    "approval_request",
    "draft_or_approval_request",
    *INTERNAL_SAFE_ACTION_TYPES,
}


def is_read_only_action(action_name: str) -> bool:
    return action_name in READ_ONLY_ACTIONS


def requires_approval_action(action_name: str, risk_level: str) -> bool:
    metadata = tool_metadata(action_name)
    if metadata:
        return bool(metadata.get("requires_approval")) or risk_level in {"medium", "high"}
    return action_name in REQUIRES_APPROVAL_ACTIONS or risk_level in {"medium", "high"}


def is_forbidden_direct_action(action_name: str) -> bool:
    return action_name in FORBIDDEN_DIRECT_ACTIONS


def tool_metadata(action_name: str) -> dict | None:
    definition = AGENT_TOOL_REGISTRY.get(action_name)
    if not definition:
        return None
    metadata = definition.get("metadata")
    if metadata:
        return metadata
    keys = {
        "risk_level",
        "effect_kind",
        "side_effect",
        "requires_approval",
        "allowed_callers",
        "idempotency_fields",
        "audit_event_type",
        "exposed_to_llm",
    }
    fallback = {key: definition.get(key) for key in keys if key in definition}
    return fallback or None


def _action_target_id(action: AgentAction) -> str | None:
    raw_decision = action.raw_decision
    target = getattr(raw_decision, "target", None)
    if target is not None and getattr(target, "canonical_id", None):
        return target.canonical_id
    return action.action_args.get("strain_id")


def _existing_workflow_pending(action: AgentAction, state: AgentRunState) -> dict | None:
    if action.action_name != "trigger_subculture_workflow":
        return None
    target_id = _action_target_id(action)
    if not target_id:
        return None
    for item in state.pending_actions or []:
        if item.get("status") != "pending":
            continue
        if "workflow" not in str(item.get("action_type") or ""):
            continue
        if str(item.get("target") or "") == str(target_id):
            return item
    return None


def evaluate_policy(action: AgentAction, state: AgentRunState) -> AgentPolicyVerdict:
    metadata = tool_metadata(action.action_name)
    if metadata and not context_allows_tool(state.request_context, action.action_name):
        return AgentPolicyVerdict(
            allowed=False,
            category="role_toolset_boundary",
            reason=f"Role {state.request_context.role if state.request_context else 'internal_scientist'} cannot use tool {action.action_name}.",
            risk_level=metadata.get("risk_level") or action.risk_level,
            requires_approval=bool(metadata.get("requires_approval")),
            forbidden=True,
            evidence={"boundary": "runtime_request_context", "tool": action.action_name},
        )
    if is_forbidden_direct_action(action.action_name):
        return AgentPolicyVerdict(
            allowed=False,
            category="block_forbidden",
            reason=f"Direct execution is forbidden for action: {action.action_name}",
            risk_level=action.risk_level,
            requires_approval=action.requires_approval,
            forbidden=True,
        )

    existing_pending = _existing_workflow_pending(action, state)
    if existing_pending:
        return AgentPolicyVerdict(
            allowed=False,
            category="existing_approval_boundary",
            reason="An existing workflow pending request already covers this target; stay at the approval boundary.",
            risk_level=action.risk_level,
            requires_approval=True,
            forbidden=False,
            evidence={
                "boundary": "existing_workflow_pending",
                "pending_id": existing_pending.get("id"),
                "target": existing_pending.get("target"),
            },
        )

    if metadata:
        effect_kind = metadata.get("effect_kind")
        risk_level = metadata.get("risk_level") or action.risk_level
        requires_approval = bool(metadata.get("requires_approval"))
        if effect_kind == "read" and not requires_approval:
            return AgentPolicyVerdict(
                allowed=True,
                category="allow_read_only",
                reason="Tool metadata marks this action as read-only.",
                risk_level=risk_level,
                requires_approval=False,
                forbidden=False,
            )
        if requires_approval or effect_kind in {"propose", "draft"}:
            return AgentPolicyVerdict(
                allowed=True,
                category="require_approval",
                reason="Tool metadata requires pending approval or frontend confirmation.",
                risk_level=risk_level,
                requires_approval=True,
                forbidden=False,
            )
        if effect_kind == "execute" and risk_level == "high":
            return AgentPolicyVerdict(
                allowed=False,
                category="block_forbidden",
                reason="High-risk execution tools cannot run directly from chat runtime.",
                risk_level=risk_level,
                requires_approval=requires_approval,
                forbidden=True,
            )

    if is_read_only_action(action.action_name) or action.action_type == "read":
        return AgentPolicyVerdict(
            allowed=True,
            category="allow_read_only",
            reason="Read-only action is allowed.",
            risk_level=action.risk_level,
            requires_approval=False,
            forbidden=False,
        )

    if action.action_name in REQUIRES_APPROVAL_ACTIONS or action.action_type in {
        "approval_request",
        "draft_or_approval_request",
    }:
        return AgentPolicyVerdict(
            allowed=True,
            category="require_approval",
            reason="Action may only create a pending request or draft in the chat path.",
            risk_level=action.risk_level,
            requires_approval=True,
            forbidden=False,
        )

    if action.action_type in INTERNAL_SAFE_ACTION_TYPES:
        return AgentPolicyVerdict(
            allowed=True,
            category="allow",
            reason="Internal non-side-effect chat control action is allowed.",
            risk_level=action.risk_level,
            requires_approval=action.requires_approval,
            forbidden=False,
        )

    if action.action_type not in KNOWN_ACTION_TYPES:
        return AgentPolicyVerdict(
            allowed=False,
            category="block_unknown_action",
            reason=f"Unknown action type cannot be executed: {action.action_type}",
            risk_level=action.risk_level,
            requires_approval=action.requires_approval,
            forbidden=False,
        )

    return AgentPolicyVerdict(
        allowed=False,
        category="block_unknown_action",
        reason=f"Unknown action cannot be executed: {action.action_name}",
        risk_level=action.risk_level,
        requires_approval=action.requires_approval,
        forbidden=False,
    )


def action_type_for_route(route_kind: str) -> str:
    if route_kind in {
        RouteKind.LAB_QUERY.value,
        RouteKind.PENDING_QUERY.value,
        RouteKind.KNOWLEDGE_QUERY.value,
        RouteKind.TOOL_INFO.value,
        RouteKind.WORKFLOW_AUDIT.value,
    }:
        return "read"
    if route_kind in {RouteKind.WRITE_ACTION.value, RouteKind.WORKFLOW.value}:
        return "approval_request"
    if route_kind == RouteKind.EMAIL.value:
        return "draft_or_approval_request"
    if route_kind == RouteKind.CHAT.value:
        return "chat"
    if route_kind == RouteKind.CLARIFICATION.value:
        return "clarification"
    if route_kind == RouteKind.PENDING_FORM.value:
        return "pending_form"
    if route_kind == RouteKind.COMPOSITE.value:
        return "composite"
    return "unknown"
