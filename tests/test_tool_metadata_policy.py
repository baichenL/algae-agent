from app.services.agent_runtime.policy import evaluate_policy, tool_metadata
from app.services.agent_runtime.state import AgentAction, AgentRunState
from app.tools.registry import AGENT_TOOL_REGISTRY


def _state():
    return AgentRunState(
        agent_run_id="policy-meta-run",
        session_id="policy-meta-session",
        user_message="test",
        conversation_history=[],
    )


def _action(name: str, action_type: str = "read", risk_level: str = "low") -> AgentAction:
    metadata = tool_metadata(name) or {}
    return AgentAction(
        action_type=action_type,
        action_name=name,
        action_args={},
        risk_level=metadata.get("risk_level") or risk_level,
        requires_approval=bool(metadata.get("requires_approval")),
    )


def test_registered_tools_have_governance_metadata():
    required = {
        "risk_level",
        "effect_kind",
        "side_effect",
        "requires_approval",
        "allowed_callers",
        "idempotency_fields",
        "audit_event_type",
        "exposed_to_llm",
    }

    for name, definition in AGENT_TOOL_REGISTRY.items():
        metadata = definition.get("metadata") or {}
        assert required <= set(metadata), name


def test_tool_metadata_allows_read_only_tool():
    verdict = evaluate_policy(_action("list_algae_strains"), _state())

    assert verdict.allowed is True
    assert verdict.category == "allow_read_only"
    assert verdict.requires_approval is False


def test_tool_metadata_requires_approval_for_write_tool():
    verdict = evaluate_policy(
        _action("update_algae_strain", action_type="approval_request"),
        _state(),
    )

    assert verdict.allowed is True
    assert verdict.category == "require_approval"
    assert verdict.requires_approval is True


def test_tool_metadata_requires_approval_for_high_risk_workflow_tool():
    verdict = evaluate_policy(
        _action("trigger_subculture_workflow", action_type="approval_request"),
        _state(),
    )

    assert verdict.allowed is True
    assert verdict.category == "require_approval"
    assert verdict.risk_level == "high"


def test_unknown_tool_falls_back_to_unknown_block():
    verdict = evaluate_policy(
        AgentAction(
            action_type="external_mutation",
            action_name="unknown_external_tool",
            action_args={},
            risk_level="high",
            requires_approval=False,
        ),
        _state(),
    )

    assert verdict.allowed is False
    assert verdict.category == "block_unknown_action"
