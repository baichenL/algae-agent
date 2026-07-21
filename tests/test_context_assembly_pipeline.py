from types import SimpleNamespace

import pytest

from app.services.agent_runtime.serialization import deserialize_runtime_state, serialize_runtime_state
from app.services.agent_runtime.state import AgentObservation, AgentRunState, AgentTerminalStatus
from app.services.context import ContextBudgetExceeded, assemble_model_input, build_status_bar
from app.services.context.compression import compress_history, trim_lane
from app.services.skills import load_skill_definition, select_skill_definitions
from scripts.run_context_eval import run_context_eval


def _snapshot(*, pending_id: int = 7, created_at: str = "2026-01-01"):
    return SimpleNamespace(
        session_id="ctx-s1",
        created_at=created_at,
        strains=[{"strain_id": "Chlorella_01", "generation_number": 4}],
        pending_actions=[{"id": pending_id, "action_type": "subculture_workflow", "status": "pending"}],
        recent_experiments=[],
        target_strain={"strain_id": "Chlorella_01"},
        selected_agent_memories=[],
        selected_user_memories=[],
    )


def _assemble(**overrides):
    values = {
        "request_kind": "workflow_write",
        "route_kind": "workflow",
        "current_user_message": "为 Chlorella_01 创建传代审批请求",
        "system_instruction": "Test system boundary.",
        "task_protocol": {"task": "Return strict JSON", "output_schema": {"status": "string"}},
        "context_snapshot": _snapshot(),
        "conversation_history": [
            {"role": "system", "content": "legacy session system"},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "为 Chlorella_01 创建传代审批请求"},
        ],
    }
    values.update(overrides)
    return assemble_model_input(**values)


def test_envelope_has_fixed_message_order_and_no_duplicate_current_turn(isolated_sqlite_db):
    envelope = _assemble()
    messages = envelope.to_messages("为 Chlorella_01 创建传代审批请求")

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "system"
    assert messages[-2]["content"].startswith("Agent status bar:")
    assert messages[-1] == {"role": "user", "content": "为 Chlorella_01 创建传代审批请求"}
    assert sum(item["content"] == messages[-1]["content"] for item in messages) == 1
    assert any(item["content"].startswith("Authoritative dynamic context:") for item in messages)
    assert any(item["content"].startswith("Selected skill procedures") for item in messages)


def test_prefix_hash_ignores_dynamic_snapshot_and_changes_with_route(isolated_sqlite_db):
    first = _assemble(context_snapshot=_snapshot(pending_id=1, created_at="2026-01-01"))
    second = _assemble(context_snapshot=_snapshot(pending_id=99, created_at="2026-07-19"))
    knowledge = _assemble(request_kind="knowledge_query", route_kind="knowledge_query")

    assert first.budget_report.prefix_hash == second.budget_report.prefix_hash
    assert first.budget_report.prefix_hash != knowledge.budget_report.prefix_hash


def test_profiles_filter_authority_slices(isolated_sqlite_db):
    state = AgentRunState("run", "s", "question", [])
    state.runtime_context = {
        "context_bundle": {
            "fact_context": {"authority": "database_current_state", "payload": {"strain_count": 1}},
            "rag_context": {"authority": "knowledge_only", "payload": {"evidence": ["ev1"]}},
            "tool_context": {"authority": "current_run_observation", "payload": {}},
            "policy_context": {"authority": "execution_boundary", "payload": {}},
        }
    }
    chat = _assemble(request_kind="chat", route_kind="chat", runtime_state=state, context_snapshot=None)
    knowledge = _assemble(request_kind="knowledge_query", route_kind="knowledge_query", runtime_state=state, context_snapshot=None)

    assert "rag_context" not in chat.dynamic_context
    assert "fact_context" not in knowledge.dynamic_context
    assert knowledge.dynamic_context["rag_context"]["authority"] == "knowledge_only"


def test_history_and_observation_compression_are_traceable(monkeypatch):
    monkeypatch.setenv("CONTEXT_RECENT_MESSAGE_COUNT", "2")
    history = [{"role": "user" if index % 2 == 0 else "assistant", "content": f"turn-{index}"} for index in range(8)]
    compressed, records = compress_history(history)

    assert len(compressed) == 3
    assert records[0].generation == 1
    assert records[0].source_refs
    assert records[0].source_hash


def test_budget_trim_preserves_safety_identifiers():
    payload = {
        "large": "x" * 5000,
        "pending_id": 42,
        "error_event_id": 91,
        "evidence_ids": ["ev-1"],
        "stop_conditions": ["approval_required"],
    }
    trimmed, records = trim_lane(payload, 80, lane="facts_policy")
    protected = trimmed["protected"]

    assert protected["pending_id"] == 42
    assert protected["error_event_id"] == 91
    assert protected["evidence_ids"] == ["ev-1"]
    assert records[0].generation == 1


def test_protected_content_over_hard_limit_stops_model_call(monkeypatch, isolated_sqlite_db):
    monkeypatch.setenv("CONTEXT_INPUT_TOKEN_BUDGET", "1000")
    monkeypatch.setenv("CONTEXT_HARD_BUDGET_RATIO", "0.80")
    with pytest.raises(ContextBudgetExceeded) as caught:
        _assemble(current_user_message="藻" * 900)
    assert caught.value.report.protected_tokens > caught.value.report.hard_limit


def test_skill_hydration_keeps_static_safety_boundary(isolated_sqlite_db):
    selected, trace = select_skill_definitions(
        route_kind="workflow",
        message="为 Chlorella_01 创建传代审批请求",
        requested_effect="execute",
        entity_mentions=["Chlorella_01"],
        max_skills=2,
    )
    definition = load_skill_definition("subculture_workflow_skill")

    assert selected[0]["name"] == "subculture_workflow_skill"
    assert selected[0]["allowed_tools"] == ["trigger_subculture_workflow"]
    assert "direct execution" in selected[0]["forbidden_actions"]
    assert definition["checksum"]
    assert any(item["selected"] for item in trace)


def test_status_bar_and_checkpoint_round_trip_new_context_state():
    state = AgentRunState("run", "session", "start workflow", [])
    state.user_goal = "start workflow"
    state.step_index = 2
    state.max_steps = 8
    state.pending_actions = [{"id": 42, "status": "pending"}]
    state.previous_observations = [
        AgentObservation(status="success", action="trigger_subculture_workflow", route_kind="workflow", output={}, pending_id=42)
    ]
    state.terminal_status = AgentTerminalStatus.WAITING_APPROVAL
    state.status_bar = build_status_bar(state, selected_skills=["subculture_workflow_skill@1"]).to_dict()
    state.compression_records = [{"record_id": "ctxcmp:1", "generation": 1}]
    state.selected_skill_definitions = [{"name": "subculture_workflow_skill", "version": 1}]
    state.skill_selection_trace = [{"name": "subculture_workflow_skill", "selected": True}]
    state.model_input_reports = [{"prompt_version": "algae-agent-context/v2"}]
    state.context_budget_report = {"estimated_input_tokens": 123}

    restored = deserialize_runtime_state(serialize_runtime_state(state))

    assert restored.status_bar["workflow_status"] == "waiting_approval"
    assert restored.status_bar["pending_ids"] == [42]
    assert restored.compression_records == state.compression_records
    assert restored.selected_skill_definitions == state.selected_skill_definitions
    assert restored.context_budget_report == state.context_budget_report


def test_context_ablation_eval_meets_safety_and_reduction_targets(isolated_sqlite_db):
    report = run_context_eval()

    assert report["status"] == "passed"
    assert report["meets_25pct_reduction_target"] is True
    assert all(item["prefix_stable_across_dynamic_change"] for item in report["results"])
    assert all(item["protected_pending_present"] for item in report["results"])
