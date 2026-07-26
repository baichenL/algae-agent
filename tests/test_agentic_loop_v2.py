from __future__ import annotations

import asyncio
import ast
import datetime
import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from app.core import database
from app.core.db import schema, scientific as scientific_db
from app.core.effects import CanonicalExecutionStatus, EffectClass
from app.core.workspaces import WorkspaceContext, workspace_scope
from app.core.workspace_storage_migration import migrate_legacy_workspace
from app.services.agent_runtime import graph_runtime
from app.services.agent_runtime.contracts_v2 import (
    ModelAction,
    ModelActionKind,
    ModelToolCall,
)
from app.services.agent_runtime.model_action_provider_v2 import ModelActionResult
from app.services.agent_runtime.serialization import (
    deserialize_runtime_state,
    serialize_agent_decision,
    serialize_runtime_state,
)
from app.services.agent_runtime.state import AgentRunState, RuntimeRequestContext
from app.services.agent_runtime.tool_execution_v2 import execute_tool_calls
from app.services.agent_runtime.tool_resolver_v2 import resolve_tools
from app.services.approval_execution_service import execute_approved_pending
from app.services.scientific.models import ExperimentCondition, ExperimentDesignSpec
from app.services.scientific.service import create_experiment_proposal
from app.core.split_storage_verifier import verify_manifest


def _state() -> AgentRunState:
    return AgentRunState(
        agent_run_id="agentic-run-1",
        session_id="agentic-session-1",
        user_message="调查 137AH 生长变慢的原因",
        conversation_history=[{"role": "user", "content": "调查 137AH 生长变慢的原因"}],
        request_context=RuntimeRequestContext(
            owner="scientist-a",
            role="scientist",
            workspace_id="shared",
            conversation_id="agentic-session-1",
        ),
        max_steps=24,
    )


def test_resolver_exposes_cross_domain_safe_tools_without_server_owned_fields(
    monkeypatch,
    isolated_sqlite_db,
):
    monkeypatch.setenv("AGENT_TOOL_LOOP_MODE", "full")
    state = _state()
    assert graph_runtime._prepare_agentic_runtime(state, "scientific_task")
    tools = {item.name: item for item in graph_runtime._resolved_tools_from_state(state)}

    assert {
        "strain_state_get",
        "experiment_history_get",
        "knowledge_evidence_search",
        "workflow_runs_list",
        "scientific_dataset_get",
        "hypothesis_ledger_update",
        "candidate_plan_record",
        "experiment_proposal_request",
    }.issubset(tools)
    assert all(
        item.effect_class
        not in {EffectClass.DOMAIN_FACT_COMMIT.value, EffectClass.EXTERNAL_ACTUATION.value}
        for item in tools.values()
    )
    for tool in tools.values():
        assert "workspace_id" not in (tool.input_schema.get("properties") or {})
        assert "agent_run_id" not in (tool.input_schema.get("properties") or {})
        assert "approval_version" not in (tool.input_schema.get("properties") or {})


def test_tool_executor_rejects_server_owned_and_field_invalid_arguments(
    monkeypatch,
    isolated_sqlite_db,
):
    monkeypatch.setenv("AGENT_TOOL_LOOP_MODE", "full")
    state = _state()
    graph_runtime._prepare_agentic_runtime(state, "scientific_task")
    envelope = graph_runtime._safety_envelope_from_state(state)
    tools = graph_runtime._resolved_tools_from_state(state)

    observations = asyncio.run(
        execute_tool_calls(
            (
                ModelToolCall(
                    "call-override",
                    "strain_state_get",
                    {"strain_id": "137AH", "workspace_id": "other"},
                ),
                ModelToolCall(
                    "call-invalid",
                    "scientific_dataset_get",
                    {},
                ),
            ),
            tools,
            envelope,
            session_id=state.session_id,
            agent_run_id=state.agent_run_id,
        )
    )

    assert observations[0].error_code == "server_owned_argument"
    assert observations[1].error_code == "invalid_arguments"
    assert observations[1].suggested_repairs


def test_model_can_repair_invalid_tool_arguments(monkeypatch, isolated_sqlite_db):
    monkeypatch.setenv("AGENT_TOOL_LOOP_MODE", "full")
    state = _state()
    graph_runtime._prepare_agentic_runtime(state, "scientific_task")
    envelope = graph_runtime._safety_envelope_from_state(state)
    tools = graph_runtime._resolved_tools_from_state(state)

    invalid = asyncio.run(
        execute_tool_calls(
            (ModelToolCall("invalid", "scientific_dataset_get", {}),),
            tools,
            envelope,
            session_id=state.session_id,
            agent_run_id=state.agent_run_id,
        )
    )[0]
    repaired = asyncio.run(
        execute_tool_calls(
            (
                ModelToolCall(
                    "repaired",
                    "scientific_dataset_get",
                    {"dataset_id": "corrected-dataset-id"},
                ),
            ),
            tools,
            envelope,
            session_id=state.session_id,
            agent_run_id=state.agent_run_id,
        )
    )[0]

    assert invalid.error_code == "invalid_arguments"
    assert repaired.error_code is None
    assert repaired.status == "not_found"


def test_model_changes_tool_after_not_found_observation(
    monkeypatch,
    isolated_sqlite_db,
):
    monkeypatch.setenv("AGENT_TOOL_LOOP_MODE", "full")
    state = _state()
    graph_runtime._prepare_agentic_runtime(state, "scientific_task")

    def scripted_provider(runtime_state, tools):
        if not runtime_state.v2_observations:
            action = ModelAction(
                kind=ModelActionKind.TOOL_CALLS,
                tool_calls=(
                    ModelToolCall("call-1", "strain_state_get", {"strain_id": "137AH"}),
                ),
            )
        elif runtime_state.v2_observations[-1]["status"] == "not_found":
            action = ModelAction(
                kind=ModelActionKind.TOOL_CALLS,
                tool_calls=(ModelToolCall("call-2", "list_algae_strains", {}),),
                reason_summary="The exact id was absent, so inspect authoritative ids.",
            )
        else:
            action = ModelAction(
                kind=ModelActionKind.FINAL_ANSWER,
                content="没有找到 137AH 的权威记录，因此未创建 proposal。",
            )
        return ModelActionResult(action=action, total_tokens=20, model="scripted")

    monkeypatch.setattr(graph_runtime, "decide_model_action", scripted_provider)
    first, _ = graph_runtime._run_agentic_model_turn(state, raw_decision=None)
    assert first.action_name == "strain_state_get"

    first_call = tuple(
        ModelToolCall(**item) for item in state.last_model_action["tool_calls"]
    )
    observations = asyncio.run(
        execute_tool_calls(
            first_call,
            graph_runtime._resolved_tools_from_state(state),
            graph_runtime._safety_envelope_from_state(state),
            session_id=state.session_id,
            agent_run_id=state.agent_run_id,
        )
    )
    state.v2_observations.extend(item.to_model_payload() for item in observations)

    second, _ = graph_runtime._run_agentic_model_turn(state, raw_decision=None)
    assert second.action_name == "list_algae_strains"
    assert second.action_name != first.action_name


def test_conflicting_authorities_remain_explicit_in_hypothesis_ledger(
    monkeypatch,
    isolated_sqlite_db,
):
    from app.services.chat import chat_service  # noqa: F401

    monkeypatch.setenv("AGENT_TOOL_LOOP_MODE", "full")
    state = _state()
    graph_runtime._prepare_agentic_runtime(state, "scientific_task")
    state.v2_observations = [
        {
            "observation_id": "obs-domain",
            "tool_call_id": "domain-read",
            "tool_name": "strain_state_get",
            "status": "success",
            "effect_class": "read",
            "authority": "domain_fact",
            "data": {"growth_status": "slower"},
            "evidence_refs": [
                {"ref_id": "e-domain", "authority": "domain_fact"}
            ],
        },
        {
            "observation_id": "obs-rag",
            "tool_call_id": "rag-read",
            "tool_name": "knowledge_evidence_search",
            "status": "success",
            "effect_class": "read",
            "authority": "rag_advisory",
            "data": {"claim": "growth should be normal"},
            "evidence_refs": [
                {"ref_id": "e-rag", "authority": "rag_advisory"}
            ],
        },
    ]

    def scripted_provider(runtime_state, tools):
        return ModelActionResult(
            action=ModelAction(
                kind=ModelActionKind.TOOL_CALLS,
                tool_calls=(
                    ModelToolCall(
                        "ledger-conflict",
                        "hypothesis_ledger_update",
                        {
                            "hypotheses": [
                                {
                                    "hypothesis_id": "h-growth",
                                    "claim": "The recent slowdown is real.",
                                    "status": "active",
                                    "supporting_evidence_refs": ["e-domain"],
                                    "contradicting_evidence_refs": ["e-rag"],
                                    "confidence": 0.55,
                                    "uncertainties": ["Authority conflict remains unresolved."],
                                    "next_best_test": "Inspect recent environment records.",
                                }
                            ],
                            "decision_summary": (
                                "Preserve both observations and investigate the conflict."
                            ),
                        },
                    ),
                ),
            ),
            total_tokens=20,
            model="scripted",
        )

    monkeypatch.setattr(graph_runtime, "decide_model_action", scripted_provider)
    decision, _ = graph_runtime._run_agentic_model_turn(state, raw_decision=None)
    graph_state = graph_runtime._base_graph_state(
        agent_run_id=state.agent_run_id,
        session_id=state.session_id,
        graph_thread_id="thread-conflict",
        user_message=state.user_message,
        max_steps=state.max_steps,
        runtime_state=serialize_runtime_state(state),
    )
    graph_state["decision"] = serialize_agent_decision(decision)

    update = asyncio.run(graph_runtime._execute_action_node(graph_state))
    restored = deserialize_runtime_state(update["runtime_state"])

    assert restored.hypotheses, json.dumps(
        restored.v2_observations[-1],
        ensure_ascii=False,
    )
    assert restored.hypotheses[0]["supporting_evidence_refs"] == ["e-domain"]
    assert restored.hypotheses[0]["contradicting_evidence_refs"] == ["e-rag"]
    assert restored.hypotheses[0]["status"] == "weakened"
    assert restored.hypotheses[0]["confidence"] < 0.55


def test_agent_cannot_claim_execution_without_canonical_state(
    monkeypatch,
    isolated_sqlite_db,
):
    monkeypatch.setenv("AGENT_TOOL_LOOP_MODE", "full")
    state = _state()
    graph_runtime._prepare_agentic_runtime(state, "scientific_task")
    response = graph_runtime._agentic_final_response(
        state,
        content="邮件已发送，硬件已执行。",
    )
    assert "不能声称" in response.natural_reply
    assert response.state_observation_refs == []


def test_budget_exhaustion_returns_partial_checkpoint_result(
    monkeypatch,
    isolated_sqlite_db,
):
    monkeypatch.setenv("AGENT_TOOL_LOOP_MODE", "full")
    state = _state()
    graph_runtime._prepare_agentic_runtime(state, "scientific_task")
    state.model_turn_count = graph_runtime._safety_envelope_from_state(
        state
    ).budgets.model_turns

    decision, response = graph_runtime._run_agentic_model_turn(
        state,
        raw_decision=None,
    )

    assert decision is None
    assert response.agent_status == "paused"
    assert response.agent_output["reason"] == "budget_exhausted:model_turns"
    assert response.agent_output["outcome_status"] == "paused"
    assert response.agent_output["budget_exhausted"] == "model_turns"


def test_simulation_failure_returns_repair_fields_for_plan_patch(
    monkeypatch,
    isolated_sqlite_db,
):
    monkeypatch.setenv("AGENT_TOOL_LOOP_MODE", "full")
    dataset = {
        "id": "dataset-capacity",
        "name": "capacity",
        "source_file": "capacity.csv",
        "content_hash": "capacity-hash",
        "mapping": {"metric_name": "od750"},
        "row_count": 1,
    }
    scientific_db.insert_dataset(
        dataset,
        [
            {
                "id": "batch-capacity",
                "condition": {"light": 100.0},
                "measurements": [
                    {
                        "elapsed_hours": 0.0,
                        "metric_name": "od750",
                        "value": 0.1,
                    }
                ],
            }
        ],
    )
    run_id = "sci-capacity"
    scientific_db.insert_scientific_run(
        {
            "id": run_id,
            "dataset_id": dataset["id"],
            "goal": {"dataset_id": dataset["id"]},
            "status": "design_ready",
            "mode": "diagnose_and_optimize",
        }
    )
    conditions = [
        ExperimentCondition(
            condition_id=f"condition-{index}",
            factors={"light": float(100 + index)},
            predicted_value=0.5,
            uncertainty=0.1,
            acquisition_score=0.5,
            role="control" if index == 0 else "candidate",
        )
        for index in range(5)
    ]
    design = ExperimentDesignSpec(
        design_id="capacity-design",
        scientific_run_id=run_id,
        target_metric="od750",
        direction="maximize",
        conditions=conditions,
        replicates=3,
        sampling_hours=[0.0, 24.0],
        required_capabilities=["measure_growth"],
    ).freeze()
    state = _state()
    graph_runtime._prepare_agentic_runtime(state, "scientific_task")

    observation = asyncio.run(
        execute_tool_calls(
            (
                ModelToolCall(
                    "simulate-capacity",
                    "candidate_design_simulate",
                    {
                        "scientific_run_id": run_id,
                        "candidate_id": "candidate-capacity",
                        "design": design.to_dict(),
                    },
                ),
            ),
            graph_runtime._resolved_tools_from_state(state),
            graph_runtime._safety_envelope_from_state(state),
            session_id=state.session_id,
            agent_run_id=state.agent_run_id,
        )
    )[0]

    assert observation.status == "simulation_failed"
    assert observation.error_code == "simulation_failed"
    assert "CAPACITY_EXCEEDED" in observation.suggested_repairs


def test_expired_pending_is_rejected_at_review_queue_and_claim(isolated_sqlite_db):
    expired = (datetime.datetime.now() - datetime.timedelta(minutes=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    pending_id = database.insert_pending_action(
        "add_strain",
        {"type": "add_strain", "data": {"strain_id": "expired-1"}},
        expires_at=expired,
    )

    review = database.review_pending_action_with_resume_job(
        pending_id,
        approve=True,
    )
    assert review["reason"] == "expired"
    assert not database.queue_pending_execution(pending_id)
    assert not database.claim_pending_execution(pending_id)
    assert database.get_pending_action(pending_id)["status"] == "expired"


def test_tampered_receipt_cannot_update_canonical_state(isolated_sqlite_db):
    pending_id = database.insert_pending_action(
        "add_strain",
        {"type": "add_strain", "data": {"strain_id": "receipt-1"}},
        execution_idempotency_key="receipt-execution-1",
    )
    review = database.review_pending_action_with_resume_job(pending_id, approve=True)
    receipt = database.create_effect_receipt(
        execution_idempotency_key="receipt-execution-1",
        workspace_id="shared",
        effect_class=EffectClass.DOMAIN_FACT_COMMIT,
        executor_identity="trusted_commit_worker",
        status=CanonicalExecutionStatus.SUCCEEDED,
        result={"status": "success"},
        pending_id=pending_id,
        approval_version=review["approval_version"],
        business_fact_changed=True,
    )
    with sqlite3.connect(isolated_sqlite_db) as conn:
        conn.execute(
            "UPDATE effect_receipts SET result_json = ? WHERE receipt_id = ?",
            (json.dumps({"status": "forged"}), receipt["receipt_id"]),
        )
    with pytest.raises(ValueError, match="receipt_hash_mismatch"):
        database.reduce_effect_receipt(receipt["receipt_id"])
    assert database.get_state_observation(receipt["receipt_id"]) is None


def test_trusted_domain_commit_is_exactly_once_across_retries(isolated_sqlite_db):
    pending_id = database.insert_pending_action(
        "add_strain",
        {
            "type": "add_strain",
            "data": {
                "strain_id": "exactly-once-1",
                "name_cn": "测试藻",
                "name_en": "Test algae",
                "generation_number": 1,
                "days_since_last_subculture": 0,
            },
        },
        execution_idempotency_key="domain-execution-exactly-once-1",
    )
    database.review_pending_action_with_resume_job(pending_id, approve=True)
    database.queue_pending_execution(pending_id)

    first = asyncio.run(execute_approved_pending(pending_id))
    second = asyncio.run(execute_approved_pending(pending_id))

    assert first["state_observation"]["canonical_status"] == "succeeded"
    assert second["idempotent"] is True
    with sqlite3.connect(isolated_sqlite_db) as conn:
        assert conn.execute(
            """
            SELECT COUNT(*) FROM domain_executions
            WHERE execution_idempotency_key = ?
            """,
            ("domain-execution-exactly-once-1",),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM algae_status WHERE strain_id = ?",
            ("exactly-once-1",),
        ).fetchone()[0] == 1


def test_v2_proposal_abstains_when_evidence_and_simulation_are_missing(
    isolated_sqlite_db,
):
    run_id = "sci-not-ready"
    scientific_db.insert_scientific_run(
        {
            "id": run_id,
            "dataset_id": "missing-dataset",
            "goal": {"dataset_id": "missing-dataset"},
            "status": "design_ready",
            "mode": "diagnose_and_optimize",
        }
    )
    design = ExperimentDesignSpec(
        design_id="design-not-ready",
        scientific_run_id=run_id,
        target_metric="od750",
        direction="maximize",
        conditions=[
            ExperimentCondition(
                condition_id="control",
                factors={"light": 100.0},
                predicted_value=0.5,
                uncertainty=0.1,
                acquisition_score=0.0,
                role="control",
            )
        ],
        replicates=3,
        sampling_hours=[0.0, 24.0],
        required_capabilities=["measure_growth"],
        evidence_citations=[],
    ).freeze()

    result = create_experiment_proposal(
        run_id,
        design,
        proposal_version=2,
        created_by_tool_call_id="proposal-call-1",
    )

    assert result["status"] == "proposal_not_ready"
    assert "current_authoritative_dataset" in result["missing_requirements"]
    assert database.list_pending_actions(status="all") == []


def test_split_store_physically_separates_tables_and_blocks_runtime_writes(
    tmp_path,
    monkeypatch,
):
    workspace = WorkspaceContext(
        id="split-test",
        name="split-test",
        db_path=str(tmp_path / "legacy.sqlite3"),
        storage_mode="split",
        control_db_path=str(tmp_path / "control.sqlite3"),
        domain_db_path=str(tmp_path / "domain.sqlite3"),
    )
    monkeypatch.setenv("ALGAE_AUTH_MODE", "test")
    with workspace_scope(workspace):
        schema.init_db(include_domain=True)
    with sqlite3.connect(workspace.control_path()) as conn:
        control_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    with sqlite3.connect(workspace.domain_path()) as conn:
        domain_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "pending_actions" in control_tables
    assert "algae_status" not in control_tables
    assert {"algae_status", "experiments", "algae_audit"}.issubset(domain_tables)
    assert "pending_actions" not in domain_tables

    monkeypatch.setenv("ALGAE_AUTH_MODE", "required")
    monkeypatch.delenv("TRUSTED_WORKER_PROCESS", raising=False)
    from app.core.db.domain_connection import connect_domain_writer

    with workspace_scope(workspace), pytest.raises(PermissionError):
        connect_domain_writer(workspace.domain_path())


def test_legacy_storage_migration_verifies_control_domain_and_primary_keys(tmp_path):
    legacy = tmp_path / "legacy.sqlite3"
    control = tmp_path / "control.sqlite3"
    domain = tmp_path / "domain.sqlite3"
    manifest = tmp_path / "manifest.json"
    with sqlite3.connect(legacy) as conn:
        conn.execute(
            "CREATE TABLE pending_actions (id INTEGER PRIMARY KEY, status TEXT)"
        )
        conn.execute(
            "CREATE TABLE algae_status (strain_id TEXT PRIMARY KEY, generation_number INTEGER)"
        )
        conn.execute("INSERT INTO pending_actions VALUES (1, 'pending')")
        conn.execute("INSERT INTO algae_status VALUES ('137AH', 4)")
        conn.commit()

    try:
        result = migrate_legacy_workspace(
            legacy_path=str(legacy),
            control_path=str(control),
            domain_path=str(domain),
            manifest_path=str(manifest),
        )
    finally:
        os.chmod(legacy, stat.S_IWRITE | stat.S_IREAD)

    assert result["storage_mode"] == "split"
    assert verify_manifest(manifest)["status"] == "passed"
    assert result["verification"]["control"]["pending_actions"]["legacy"] == (
        result["verification"]["control"]["pending_actions"]["split"]
    )
    assert result["verification"]["domain"]["algae_status"]["legacy"] == (
        result["verification"]["domain"]["algae_status"]["split"]
    )
    assert (
        result["verification"]["domain"]["algae_status"]["legacy"][
            "primary_key_count"
        ]
        == 1
    )
    with sqlite3.connect(control) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM pending_actions"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'algae_status'"
        ).fetchone()[0] == 0
    with sqlite3.connect(domain) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM algae_status"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'pending_actions'"
        ).fetchone()[0] == 0


def test_safe_tool_handlers_do_not_import_actuation_or_fact_commit_dependencies():
    handler_files = list(Path("app/tools").glob("*_tool_handlers.py"))
    handler_files.append(Path("app/tools/email_tool.py"))
    forbidden_modules = {
        "app.services.email.email_service",
        "app.services.workflows.workflow_execution_service",
        "app.services.approval_execution_service",
    }
    forbidden_names = {
        "send_email",
        "execute_approved_pending",
        "execute_approved_pending_action",
        "add_algae_strain",
        "update_algae_status",
        "delete_algae_strain",
    }
    violations: list[str] = []
    for path in handler_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in forbidden_modules:
                    violations.append(f"{path}:{node.module}")
                for alias in node.names:
                    if alias.name in forbidden_names:
                        violations.append(f"{path}:{alias.name}")
    assert violations == []
