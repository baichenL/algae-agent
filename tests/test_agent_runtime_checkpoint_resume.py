import asyncio
import ast
import inspect
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.core import database
from app.schemas.algae import ChatResponse
from app.services.agent_runtime import approval_resume
from app.services.agent_runtime.checkpoint import graph_thread_id_for_run
from app.services.agent_runtime.graph_runtime import (
    _await_approval_node,
    _base_graph_state,
    build_agent_runtime_graph,
)
from app.services.agent_runtime.serialization import (
    ACTION_SCHEMA_VERSION,
    DECISION_SCHEMA_VERSION,
    GRAPH_DEFINITION_VERSION,
    STATE_SCHEMA_VERSION,
    deserialize_runtime_state,
    serialize_runtime_state,
    validate_graph_versions,
)
from app.services.agent_runtime.state import AgentDecision, AgentRunState
from app.services.intent.routing_models import (
    EntityRef,
    ReasonCode,
    RiskLevel,
    RouteCandidate,
    RouteKind,
)
from app.services.strains import strain_service


def _rows(db_path: Path, sql: str, params=()):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _assert_primitive_tree(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _assert_primitive_tree(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_primitive_tree(item)
        return
    raise AssertionError(f"non-serializable value in graph state: {type(value)!r}")


def test_sqlite_checkpointer_compiles_interrupts_and_resumes(tmp_path):
    async def run():
        conn = await aiosqlite.connect(tmp_path / "checkpoints.sqlite")
        saver = AsyncSqliteSaver(conn)
        await saver.setup()

        graph = StateGraph(dict)
        graph.add_node("Await", lambda state: {"approval": interrupt({"pending_id": 1})})
        graph.add_edge(START, "Await")
        graph.add_edge("Await", END)
        app = graph.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "agent-run:sqlite-resume"}}
        first = await app.ainvoke({"started": True}, config=config)
        assert "__interrupt__" in first

        second = await app.ainvoke(Command(resume={"decision": "approved"}), config=config)
        assert second["approval"] == {"decision": "approved"}
        await conn.close()

    asyncio.run(run())


def test_sqlite_checkpoint_close_reopen_restores_nested_entity_types(tmp_path):
    async def run():
        checkpoint_path = tmp_path / "nested-entity-checkpoints.sqlite"
        entity = EntityRef(
            entity_type="strain",
            canonical_id="Chlamydomonas_01",
            mention="Chlamydomonas_01",
            source="explicit_id",
            exists=True,
        )
        candidate = RouteCandidate(
            kind=RouteKind.SCIENTIFIC_TASK,
            reason_code=ReasonCode.SCIENTIFIC_TASK_MATCHED,
            risk_level=RiskLevel.LOW,
            evidence="two candidate request",
            entity_options=(entity,),
        )
        runtime = AgentRunState(
            agent_run_id="sqlite-nested-run",
            session_id="sqlite-nested-session",
            user_message="generate two candidates",
            conversation_history=[],
        )
        runtime.steps = []
        runtime.runtime_context["routing_decision"] = {
            "candidate": candidate,
        }
        runtime.loop_plan = [
            AgentDecision(
                route_kind="scientific_task",
                action_type="read",
                action_name="scientific_dataset_get",
                action_args={"strain_id": "Chlamydomonas_01"},
                risk_level="low",
                requires_approval=False,
                missing_fields=[],
                can_continue=True,
                reason="checkpoint test",
                raw_decision=SimpleNamespace(
                    kind=RouteKind.SCIENTIFIC_TASK,
                    target=entity,
                    entity_options=(entity,),
                    candidates=(candidate,),
                ),
            )
        ]

        graph = StateGraph(dict)
        graph.add_node("persist", lambda state: state)
        graph.add_edge(START, "persist")
        graph.add_edge("persist", END)
        config = {"configurable": {"thread_id": "sqlite-nested-thread"}}

        first_connection = await aiosqlite.connect(checkpoint_path)
        first_saver = AsyncSqliteSaver(first_connection)
        await first_saver.setup()
        first_app = graph.compile(checkpointer=first_saver)
        await first_app.ainvoke(
            {"runtime_state": serialize_runtime_state(runtime)},
            config=config,
        )
        await first_connection.close()

        second_connection = await aiosqlite.connect(checkpoint_path)
        second_saver = AsyncSqliteSaver(second_connection)
        second_app = graph.compile(checkpointer=second_saver)
        snapshot = await second_app.aget_state(config)
        restored = deserialize_runtime_state(snapshot.values["runtime_state"])
        await second_connection.close()

        raw = restored.loop_plan[0].raw_decision
        assert raw.target.canonical_id == "Chlamydomonas_01"
        assert raw.entity_options[0].canonical_id == "Chlamydomonas_01"
        assert raw.candidates[0].entity_options[0].canonical_id == "Chlamydomonas_01"

    asyncio.run(run())


def test_agent_graph_compiles_with_sqlite_checkpointer(tmp_path):
    async def run():
        conn = await aiosqlite.connect(tmp_path / "agent-checkpoints.sqlite")
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        graph = build_agent_runtime_graph(checkpointer=saver)
        assert graph.get_graph().nodes
        await conn.close()

    asyncio.run(run())


def test_graph_thread_id_is_per_agent_run():
    assert graph_thread_id_for_run("run-a") == "agent-run:run-a"
    assert graph_thread_id_for_run("run-b") == "agent-run:run-b"
    assert graph_thread_id_for_run("run-a") != graph_thread_id_for_run("run-b")


def test_graph_state_versions_and_primitive_serialization():
    state = AgentRunState(
        agent_run_id="run-state-1",
        session_id="session-state-1",
        user_message="hello",
        conversation_history=[{"role": "user", "content": "hello"}],
    )
    state.final_response = ChatResponse(
        status="success",
        session_id="session-state-1",
        agent_output={"action": "noop"},
        natural_reply="ok",
    )
    graph_state = _base_graph_state(
        agent_run_id=state.agent_run_id,
        session_id=state.session_id,
        graph_thread_id=graph_thread_id_for_run(state.agent_run_id),
        user_message=state.user_message,
        max_steps=1,
        runtime_state=serialize_runtime_state(state),
    )

    assert graph_state["state_schema_version"] == STATE_SCHEMA_VERSION
    assert graph_state["graph_definition_version"] == GRAPH_DEFINITION_VERSION
    assert graph_state["decision_schema_version"] == DECISION_SCHEMA_VERSION
    assert graph_state["action_schema_version"] == ACTION_SCHEMA_VERSION
    assert validate_graph_versions(graph_state)
    _assert_primitive_tree(graph_state)


def test_incompatible_checkpoint_versions_are_rejected():
    graph_state = {
        "state_schema_version": 0,
        "graph_definition_version": GRAPH_DEFINITION_VERSION,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
    assert not validate_graph_versions(graph_state)


def test_await_approval_first_effective_operation_is_interrupt():
    source = inspect.getsource(_await_approval_node)
    tree = ast.parse(source)
    first_stmt = tree.body[0].body[0]
    assert isinstance(first_stmt, ast.Assign)
    assert isinstance(first_stmt.value, ast.Call)
    assert getattr(first_stmt.value.func, "id", None) == "interrupt"


def test_execution_and_domain_dedupe_keys_are_separate(isolated_sqlite_db):
    payload = {"type": "add_strain", "data": {"strain_id": "Dedup_01"}}
    first = database.insert_pending_action(
        "add_strain",
        payload,
        agent_run_id="run-1",
        graph_thread_id="agent-run:run-1",
        execution_idempotency_key="exec-run-1-step-1",
        domain_dedupe_key="domain-add-dedup-01",
    )
    same_execution = database.insert_pending_action(
        "add_strain",
        payload,
        agent_run_id="run-1",
        graph_thread_id="agent-run:run-1",
        execution_idempotency_key="exec-run-1-step-1",
        domain_dedupe_key="domain-add-dedup-01-other",
    )
    same_domain = database.insert_pending_action(
        "add_strain",
        payload,
        agent_run_id="run-2",
        graph_thread_id="agent-run:run-2",
        execution_idempotency_key="exec-run-2-step-1",
        domain_dedupe_key="domain-add-dedup-01",
    )

    assert same_execution == first
    assert same_domain == first


def test_review_creates_resume_job_and_is_cas_idempotent(isolated_sqlite_db):
    pending_id = database.insert_pending_action(
        "add_strain",
        {"type": "add_strain", "data": {"strain_id": "Approve_01"}},
        agent_run_id="run-approve-1",
        graph_thread_id="agent-run:run-approve-1",
        execution_idempotency_key="exec-approve-1",
        domain_dedupe_key="domain-approve-1",
    )

    first = database.review_pending_action_with_resume_job(pending_id, approve=True)
    second = database.review_pending_action_with_resume_job(pending_id, approve=True)
    denied_after_approve = database.review_pending_action_with_resume_job(pending_id, approve=False)

    assert first["status"] == "success"
    assert first["approval_version"] == 1
    assert first["job_id"]
    assert second["status"] == "success"
    assert second["idempotent"] is True
    assert second["job_id"] == first["job_id"]
    assert denied_after_approve["status"] == "error"
    assert denied_after_approve["reason"] == "already_reviewed"

    jobs = _rows(isolated_sqlite_db, "SELECT * FROM approval_resume_jobs")
    assert len(jobs) == 1
    assert jobs[0]["graph_thread_id"] == "agent-run:run-approve-1"


def test_recovery_scan_resumes_pending_jobs_with_same_thread_id(monkeypatch, isolated_sqlite_db):
    pending_id = database.insert_pending_action(
        "add_strain",
        {"type": "add_strain", "data": {"strain_id": "Recover_01"}},
        agent_run_id="run-recover-1",
        graph_thread_id="agent-run:run-recover-1",
        execution_idempotency_key="exec-recover-1",
        domain_dedupe_key="domain-recover-1",
    )
    review = database.review_pending_action_with_resume_job(pending_id, approve=True)
    calls = []

    class FakeGraph:
        async def ainvoke(self, command, config):
            calls.append((command, config))
            return {"status": "resumed"}

    monkeypatch.setattr(approval_resume, "get_compiled_graph", lambda: FakeGraph())
    results = asyncio.run(approval_resume.process_pending_resume_jobs())

    assert results[0]["status"] == "success"
    assert calls[0][1] == {"configurable": {"thread_id": "agent-run:run-recover-1"}}
    jobs = _rows(isolated_sqlite_db, "SELECT * FROM approval_resume_jobs WHERE id = ?", (review["job_id"],))
    assert jobs[0]["status"] == "succeeded"
    pending = database.get_pending_action(pending_id)
    assert pending["resume_completed_at"]


def test_failed_resume_job_is_retryable(monkeypatch, isolated_sqlite_db):
    pending_id = database.insert_pending_action(
        "add_strain",
        {"type": "add_strain", "data": {"strain_id": "Retry_01"}},
        agent_run_id="run-retry-1",
        graph_thread_id="agent-run:run-retry-1",
        execution_idempotency_key="exec-retry-1",
        domain_dedupe_key="domain-retry-1",
    )
    review = database.review_pending_action_with_resume_job(pending_id, approve=True)
    attempts = {"count": 0}

    class FlakyGraph:
        async def ainvoke(self, command, config):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary")
            return {"status": "resumed"}

    monkeypatch.setattr(approval_resume, "get_compiled_graph", lambda: FlakyGraph())
    first = asyncio.run(approval_resume.process_pending_resume_jobs())
    second = asyncio.run(approval_resume.process_pending_resume_jobs())

    assert first[0]["status"] == "error"
    assert second[0]["status"] == "success"
    jobs = _rows(isolated_sqlite_db, "SELECT * FROM approval_resume_jobs WHERE id = ?", (review["job_id"],))
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["attempt_count"] == 2


def test_execute_approved_action_uses_frozen_pending_payload(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(strain_service, "append_decision_event", lambda event: None)
    pending_id = strain_service.create_pending_add_strain(
        {
            "strain_id": "Frozen_01",
            "name_cn": "冻结参数",
            "name_en": "Frozen payload",
            "generation_number": 3,
            "days_since_last_subculture": 4,
        },
        agent_run_id="run-frozen-1",
        graph_thread_id="agent-run:run-frozen-1",
        execution_idempotency_key="exec-frozen-1",
        domain_dedupe_key="domain-frozen-1",
    )
    database.review_pending_action_with_resume_job(pending_id, approve=True)

    result = asyncio.run(strain_service.execute_approved_pending_action(pending_id))
    strain = database.get_algae_status("Frozen_01")

    assert result["status"] == "success"
    assert strain["name_en"] == "Frozen payload"
    assert strain["generation_number"] == 3
    assert database.get_algae_status("Forged_01") is None
