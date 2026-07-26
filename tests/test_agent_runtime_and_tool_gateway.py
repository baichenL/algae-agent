import asyncio
import json
from types import SimpleNamespace

import pytest

from app.core.db.connection import connect
from app.schemas.algae import ChatRequest, ChatResponse
from app.services.agent_runtime.context import build_agent_context
from app.services.agent_runtime.state import AgentObservation, AgentRunState
from app.services.agent_runtime.llm_replan_provider import (
    LlmObservationAssessment,
    LlmReplanCandidate,
    LlmReplanSuggestion,
)
from app.services.agent_runtime import replanning
from app.services.chat import chat_service, response_builder
from app.services.intent import dispatcher
from app.services.intent.routing_models import (
    ChatDecision,
    CompositeDecision,
    EmailDecision,
    EntityRef,
    KnowledgeDecision,
    QueryDecision,
    ReasonCode,
    RiskLevel,
    RouteKind,
    WorkflowDecision,
    WriteDecision,
)
from app.tools.executor import ToolExecutionContext, execute_registered_tool
from app.tools.registry import AGENT_TOOL_REGISTRY

from tests.conftest import context


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with connect(row_factory=True) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


def _event_payload(agent_run_id: str, event_type: str) -> dict:
    rows = _rows(
        "SELECT payload_json FROM agent_run_events WHERE agent_run_id = ? AND event_type = ?",
        (agent_run_id, event_type),
    )
    assert rows
    return json.loads(rows[0]["payload_json"])


def _event_payloads(agent_run_id: str, event_type: str) -> list[dict]:
    return [
        json.loads(row["payload_json"])
        for row in _rows(
            "SELECT payload_json FROM agent_run_events WHERE agent_run_id = ? AND event_type = ?",
            (agent_run_id, event_type),
        )
    ]


def _event_count(agent_run_id: str, event_type: str) -> int:
    rows = _rows(
        "SELECT COUNT(*) AS count FROM agent_run_events WHERE agent_run_id = ? AND event_type = ?",
        (agent_run_id, event_type),
    )
    return rows[0]["count"]


def test_unregistered_tool_records_error_event(isolated_sqlite_db):
    result = asyncio.run(
        execute_registered_tool(
            "missing_tool",
            {"session_id": "tool-s1", "agent_run_id": "run-tool-1"},
        )
    )

    assert result["status"] == "error"
    assert result["error_event_id"]

    events = _rows("SELECT * FROM agent_error_events WHERE agent_run_id = ?", ("run-tool-1",))
    assert events[0]["layer"] == "tool_calling"
    assert events[0]["error_type"] == "ToolNotRegistered"


def test_tool_gateway_blocks_disallowed_caller(isolated_sqlite_db):
    result = asyncio.run(
        execute_registered_tool(
            "add_algae_strain",
            {
                "strain_id": "Spirulina_03",
                "name_cn": "spirulina",
                "name_en": "Spirulina platensis",
                "session_id": "tool-caller-s1",
                "agent_run_id": "run-tool-caller-1",
            },
            context=ToolExecutionContext(caller="rag_runtime", session_id="tool-caller-s1", agent_run_id="run-tool-caller-1"),
        )
    )

    assert result["status"] == "blocked"
    events = _rows("SELECT * FROM agent_error_events WHERE agent_run_id = ?", ("run-tool-caller-1",))
    assert events[0]["error_type"] == "ToolCallerNotAllowed"


def test_tool_gateway_warns_when_metadata_is_missing(isolated_sqlite_db):
    AGENT_TOOL_REGISTRY["metadata_missing_tool"] = {
        "schema": {"name": "metadata_missing_tool"},
        "handler": lambda args: {
            "action": "metadata_missing_tool",
            "status": "success",
            "response_payload": {"action": "metadata_missing_tool", "status": "success"},
            "memory_text": "ok",
        },
    }
    try:
        result = asyncio.run(
            execute_registered_tool(
                "metadata_missing_tool",
                {"session_id": "tool-meta-s1", "agent_run_id": "run-tool-meta-1"},
            )
        )
    finally:
        AGENT_TOOL_REGISTRY.pop("metadata_missing_tool", None)

    assert result["status"] == "success"
    warning = _event_payload("run-tool-meta-1", "tool_metadata_warning")
    assert warning["warning"] == "metadata_missing"


def test_email_chat_route_uses_tool_gateway(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: EmailDecision(
        reason_code=ReasonCode.EMAIL_MATCHED,
        risk_level=RiskLevel.MEDIUM,
        explanation="email test",
    ))
    captured = {}

    async def fake_tool_executor(tool_name, function_args, allow_high_risk=False):
        captured["tool_name"] = tool_name
        captured["function_args"] = function_args
        return {
            "response_payload": {
                "action": "email_draft",
                "status": "success",
                "message": "draft ok",
                "draft": {"subject": "s", "body": "b", "recipients": []},
                "requires_confirmation": True,
            },
            "memory_text": "draft ok",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="please email lab about Chlamydomonas_01",
        session_id="email-gw-s1",
    )))

    assert response.agent_output["action"] == "email_draft"
    assert captured["tool_name"] == "email_draft"
    assert captured["function_args"]["session_id"] == "email-gw-s1"
    assert captured["function_args"]["agent_run_id"]


def test_chat_creates_agent_run_and_events(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-lifecycle-1")

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="你有哪些工具？", session_id="run-s1")))

    assert response.agent_output["action"] == "tool_info"
    runs = _rows("SELECT * FROM agent_runs WHERE id = ?", ("run-lifecycle-1",))
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["final_route"] == "tool_info"

    event_types = {
        row["event_type"]
        for row in _rows("SELECT event_type FROM agent_run_events WHERE agent_run_id = ?", ("run-lifecycle-1",))
    }
    assert {"run_started", "context_built", "routed", "response_created", "run_finished"} <= event_types
    assert {
        "agent_step_started",
        "agent_decision_made",
        "agent_policy_evaluated",
        "agent_action_started",
        "agent_observation_collected",
        "agent_step_finished",
    } <= event_types
    policy = _event_payload("run-lifecycle-1", "agent_policy_evaluated")["policy"]
    assert policy["allowed"] is True
    assert policy["category"] == "allow_read_only"
    context_payload = _event_payload("run-lifecycle-1", "context_built")
    assert context_payload["runtime_context"]["user_goal"]
    assert context_payload["runtime_context"]["step_index"] == 1
    assert context_payload["runtime_context"]["previous_observation_count"] == 0


def test_agent_context_tracks_runtime_state_for_next_decision(isolated_sqlite_db):
    state = AgentRunState(
        agent_run_id="ctx-run-1",
        session_id="ctx-s1",
        user_message="总结 TAP 培养基资料",
        conversation_history=[
            {"role": "system", "content": "test"},
            {"role": "user", "content": "总结 TAP 培养基资料"},
        ],
    )
    state.step_index = 2
    state.missing_fields = ["strain_id"]
    state.policy_history.append({"category": "allow_read_only", "allowed": True})
    state.previous_observations.append(
        AgentObservation(
            status="success",
            action="rag_answer",
            route_kind="knowledge_query",
            output={"answer": {"conclusion": "TAP is ok", "sources": ["doc-1", "doc-2"]}},
            natural_reply="TAP is ok",
        )
    )

    deps = SimpleNamespace(
        build_context_snapshot=lambda session_id: context(
            strains=[
                {
                    "strain_id": "Chlorella_01",
                    "name_cn": "小球藻",
                    "name_en": "Chlorella",
                    "generation_number": 3,
                    "subculture_due": False,
                }
            ],
            pending_actions=[
                {
                    "id": 9,
                    "action_type": "workflow_subculture",
                    "status": "pending",
                    "requester": "agent_tool",
                    "payload": {"data": {"strain_id": "Chlorella_01"}},
                }
            ],
        )
    )

    result = build_agent_context(state, deps)

    assert result.snapshot is state.context_snapshot
    assert state.user_goal == "总结 TAP 培养基资料"
    assert state.current_db_snapshot["strain_count"] == 1
    assert state.runtime_context["previous_observations"][0]["action"] == "rag_answer"
    assert state.runtime_context["missing_fields"] == ["strain_id"]
    assert state.runtime_context["policy_history"][0]["category"] == "allow_read_only"
    assert state.rag_evidence == [{"action": "rag_answer", "conclusion": "TAP is ok", "source_count": 2}]
    assert state.workflow_status["pending_workflow_count"] == 1
    assert result.event_payload["runtime_context"]["previous_observation_count"] == 1


def test_agent_observation_wraps_pending_response(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-pending-observation-1")
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: WriteDecision(
        reason_code=ReasonCode.WRITE_ACTION_MATCHED,
        risk_level=RiskLevel.MEDIUM,
        explanation="write test",
        operation="add",
        tool_name="add_algae_strain",
        arguments={"strain_id": "Spirulina_02", "name_cn": "spirulina", "name_en": "Spirulina platensis"},
    ))

    async def fake_tool_executor(tool_name, function_args, allow_high_risk=False):
        return {
            "response_payload": {
                "action": "add_strain",
                "status": "pending",
                "pending_id": 7,
                "require_confirmation": True,
            },
            "memory_text": "pending 7",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="add strain", session_id="pending-obs-s1")))

    assert response.agent_output["pending_id"] == 7
    runs = _rows("SELECT * FROM agent_runs WHERE id = ?", ("run-pending-observation-1",))
    assert runs[0]["status"] == "waiting_approval"
    events = _rows(
        "SELECT payload_json FROM agent_run_events WHERE agent_run_id = ? AND event_type = ?",
        ("run-pending-observation-1", "agent_observation_collected"),
    )
    payload = json.loads(events[0]["payload_json"])
    observation = payload["observation"]
    assert observation["status"] == "pending"
    assert observation["action"] == "add_strain"
    assert observation["pending_id"] == 7
    policy = _event_payload("run-pending-observation-1", "agent_policy_evaluated")["policy"]
    assert policy["allowed"] is True
    assert policy["category"] == "require_approval"
    assert policy["requires_approval"] is True
    assert _event_count("run-pending-observation-1", "agent_step_started") == 1
    stopped = _event_payload("run-pending-observation-1", "agent_loop_stopped")
    assert stopped["directive"]["reason"] == "terminal_status:waiting_approval"


def test_agent_observation_wraps_rag_response(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-rag-observation-1")
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: KnowledgeDecision(
        reason_code=ReasonCode.KNOWLEDGE_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="rag test",
    ))

    def fake_rag_handler(session_id, conversation_history, message, agent_run_id=None):
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={"action": "rag_answer", "answer": {"conclusion": "ok"}},
            natural_reply="rag ok",
        )

    monkeypatch.setattr(dispatcher, "handle_rag_intent", fake_rag_handler)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="what is TAP", session_id="rag-obs-s1")))

    assert response.agent_output["action"] == "rag_answer"
    events = _rows(
        "SELECT payload_json FROM agent_run_events WHERE agent_run_id = ? AND event_type = ?",
        ("run-rag-observation-1", "agent_observation_collected"),
    )
    payload = json.loads(events[0]["payload_json"])
    observation = payload["observation"]
    assert observation["status"] == "success"
    assert observation["action"] == "rag_answer"
    assert observation["route_kind"] == "knowledge_query"


def test_low_risk_composite_runs_multiple_agent_steps(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-loop-composite-1")
    pending_step = QueryDecision(
        kind=RouteKind.PENDING_QUERY,
        reason_code=ReasonCode.PENDING_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="pending step",
        query_type="pending",
        source_text="list pending",
    )
    rag_step = KnowledgeDecision(
        reason_code=ReasonCode.KNOWLEDGE_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="rag step",
        source_text="summarize TAP",
    )
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: CompositeDecision(
        reason_code=ReasonCode.COMPOSITE_PLAN,
        risk_level=RiskLevel.LOW,
        explanation="read-only composite",
        steps=(pending_step, rag_step),
    ))

    def fake_rag_handler(session_id, conversation_history, message, agent_run_id=None):
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={"action": "rag_answer", "status": "success", "answer": {"conclusion": message}},
            natural_reply=f"rag: {message}",
        )

    monkeypatch.setattr(dispatcher, "handle_rag_intent", fake_rag_handler)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="pending and TAP", session_id="loop-c1")))

    assert response.agent_output["action"] == "composite_result"
    assert response.agent_output["step_count"] == 2
    assert [step["route_kind"] for step in response.agent_output["steps"]] == [
        "pending_query",
        "knowledge_query",
    ]
    assert _event_count("run-loop-composite-1", "agent_step_started") == 2
    assert _event_count("run-loop-composite-1", "agent_loop_plan_created") == 1
    assert _event_count("run-loop-composite-1", "agent_loop_continue") == 1
    assert _event_count("run-loop-composite-1", "agent_loop_finalized") == 1


def test_single_read_only_query_stops_after_one_step(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-single-readonly-1")
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: QueryDecision(
        kind=RouteKind.LAB_QUERY,
        reason_code=ReasonCode.LAB_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="single query",
        query_type="strain_status",
    ))

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="list strains", session_id="single-read-s1")))

    assert response.agent_output["action"] in {"strain_status", "list_strains", "query_strains"}
    assert _event_count("run-single-readonly-1", "agent_step_started") == 1
    assert _event_count("run-single-readonly-1", "agent_loop_continue") == 0
    stopped = _event_payload("run-single-readonly-1", "agent_loop_stopped")
    assert stopped["directive"]["reason"] == "no_loop_plan"


def test_agent_runtime_error_records_error_event(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-error-observation-1")

    def fail_context(session_id):
        raise RuntimeError("context boom")

    monkeypatch.setattr(chat_service, "build_context_snapshot", fail_context)

    with pytest.raises(RuntimeError, match="context boom"):
        asyncio.run(chat_service.handle_chat(ChatRequest(message="hello", session_id="error-obs-s1")))

    runs = _rows("SELECT * FROM agent_runs WHERE id = ?", ("run-error-observation-1",))
    assert runs[0]["status"] == "failed"
    events = _rows("SELECT * FROM agent_error_events WHERE agent_run_id = ?", ("run-error-observation-1",))
    assert events[0]["layer"] == "chat_service"
    assert events[0]["error_type"] == "RuntimeError"


def test_workflow_policy_allows_pending_request_only(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-workflow-policy-1")
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: WorkflowDecision(
        reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
        risk_level=RiskLevel.HIGH,
        explanation="workflow test",
        target=EntityRef("strain", "Chlorella_01", "Chlorella_01", "database_id", True),
    ))
    captured = {}

    async def fake_tool_executor(tool_name, function_args, allow_high_risk=False):
        from app.services.strains import strain_service

        pending_id = strain_service.create_pending_workflow_subculture(
            function_args["strain_id"],
            requester="agent_tool",
            session_id=function_args.get("session_id"),
            agent_run_id=function_args.get("agent_run_id"),
            source_message=function_args.get("source_message"),
        )
        captured["tool_name"] = tool_name
        captured["allow_high_risk"] = allow_high_risk
        return {
            "response_payload": {
                "action": "workflow_subculture",
                "status": "pending",
                "pending_id": pending_id,
                "strain_id": "Chlorella_01",
                "require_confirmation": True,
            },
            "memory_text": "workflow pending",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="run workflow", session_id="workflow-policy-s1")))

    assert response.agent_output["status"] == "pending"
    assert response.agent_output["pending_id"]
    assert captured == {"tool_name": "trigger_subculture_workflow", "allow_high_risk": True}
    policy = _event_payload("run-workflow-policy-1", "agent_policy_evaluated")["policy"]
    assert policy["category"] == "require_approval"
    assert policy["risk_level"] == "high"


def test_forbidden_action_is_blocked_before_dispatch(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-forbidden-policy-1")
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: SimpleNamespace(
        kind=RouteKind.CHAT,
        reason_code=ReasonCode.DEFAULT_CHAT,
        risk_level=RiskLevel.HIGH,
        explanation="forbidden test",
        target=None,
        candidates=(SimpleNamespace(kind=RouteKind.CHAT, entity_options=()),),
        tool_name="commit_db_change",
        arguments={},
    ))

    async def fail_dispatch(*args, **kwargs):
        raise AssertionError("forbidden action must not reach dispatcher")

    monkeypatch.setattr(chat_service, "dispatch_routing_decision", fail_dispatch)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="commit db", session_id="forbidden-policy-s1")))

    assert response.agent_output["status"] == "blocked"
    assert response.agent_output["action"] == "commit_db_change"
    policy = _event_payload("run-forbidden-policy-1", "agent_policy_evaluated")["policy"]
    assert policy["allowed"] is False
    assert policy["category"] == "block_forbidden"
    event_types = {
        row["event_type"]
        for row in _rows("SELECT event_type FROM agent_run_events WHERE agent_run_id = ?", ("run-forbidden-policy-1",))
    }
    assert "agent_action_blocked" in event_types
    assert "agent_action_started" not in event_types
    assert _event_count("run-forbidden-policy-1", "agent_step_started") == 1
    stopped = _event_payload("run-forbidden-policy-1", "agent_loop_stopped")
    assert stopped["directive"]["reason"] == "terminal_status:blocked"


def test_unknown_action_is_blocked_before_dispatch(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-unknown-policy-1")
    unknown_kind = SimpleNamespace(value="alien_route")
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: SimpleNamespace(
        kind=unknown_kind,
        reason_code=SimpleNamespace(value="unknown_route"),
        risk_level=RiskLevel.LOW,
        explanation="unknown test",
        target=None,
        candidates=(SimpleNamespace(kind=unknown_kind, entity_options=()),),
        arguments={},
    ))

    async def fail_dispatch(*args, **kwargs):
        raise AssertionError("unknown action must not reach dispatcher")

    monkeypatch.setattr(chat_service, "dispatch_routing_decision", fail_dispatch)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="unknown", session_id="unknown-policy-s1")))

    assert response.agent_output["status"] == "blocked"
    policy = _event_payload("run-unknown-policy-1", "agent_policy_evaluated")["policy"]
    assert policy["allowed"] is False
    assert policy["category"] == "block_unknown_action"
    assert _event_count("run-unknown-policy-1", "agent_step_started") == 1


def test_loop_stops_on_duplicate_action_signature(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-loop-duplicate-1")
    first_step = QueryDecision(
        kind=RouteKind.PENDING_QUERY,
        reason_code=ReasonCode.PENDING_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="pending step",
        query_type="pending",
        source_text="list pending",
    )
    second_step = QueryDecision(
        kind=RouteKind.PENDING_QUERY,
        reason_code=ReasonCode.PENDING_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="duplicate pending step",
        query_type="pending",
        source_text="list pending again",
    )
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: CompositeDecision(
        reason_code=ReasonCode.COMPOSITE_PLAN,
        risk_level=RiskLevel.LOW,
        explanation="duplicate composite",
        steps=(first_step, second_step),
    ))

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="pending twice", session_id="loop-dup-s1")))

    assert response.agent_output["action"] == "composite_result"
    assert response.agent_output["step_count"] == 1
    assert _event_count("run-loop-duplicate-1", "agent_step_started") == 1
    stopped = _event_payload("run-loop-duplicate-1", "agent_loop_stopped")
    assert stopped["directive"]["reason"] == "duplicate_action"


def test_composite_with_high_risk_child_runs_as_guarded_runtime_step(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-high-risk-composite-1")
    target = EntityRef("strain", "Chlorella_01", "Chlorella_01", "database_id", True)
    workflow_step = WorkflowDecision(
        reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
        risk_level=RiskLevel.HIGH,
        explanation="workflow child",
        source_text="run workflow",
        target=target,
    )
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: CompositeDecision(
        reason_code=ReasonCode.COMPOSITE_PLAN,
        risk_level=RiskLevel.HIGH,
        explanation="high-risk composite",
        steps=(workflow_step,),
    ))

    async def fake_tool_executor(tool_name, function_args, allow_high_risk=False):
        from app.services.strains import strain_service

        pending_id = strain_service.create_pending_workflow_subculture(
            function_args["strain_id"],
            requester="agent_tool",
            session_id=function_args.get("session_id"),
            agent_run_id=function_args.get("agent_run_id"),
            source_message=function_args.get("source_message"),
        )
        return {
            "response_payload": {
                "action": "workflow_subculture",
                "status": "pending",
                "pending_id": pending_id,
                "strain_id": "Chlorella_01",
                "require_confirmation": True,
            },
            "memory_text": "workflow pending",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="workflow and more", session_id="hi-comp-s1")))

    assert response.agent_output["action"] == "workflow_subculture"
    assert response.agent_output["pending_id"]
    assert _event_count("run-high-risk-composite-1", "agent_step_started") == 1
    assert _event_count("run-high-risk-composite-1", "agent_loop_plan_created") == 1
    plan = _event_payload("run-high-risk-composite-1", "agent_plan_created")["plan"]
    assert plan["mode"] == "composite_guarded"


def test_dynamic_replan_creates_workflow_pending_for_single_due_strain(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-dynamic-replan-single-1")
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context(
        strains=[
            {
                "strain_id": "Chlorella_01",
                "name_cn": "chlorella",
                "name_en": "Chlorella vulgaris",
                "generation_number": 8,
                "days_since_last_subculture": 9,
            }
        ]
    ))
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: QueryDecision(
        kind=RouteKind.LAB_QUERY,
        reason_code=ReasonCode.LAB_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="find due strains first",
        query_type="due_subculture",
    ))

    async def fake_tool_executor(tool_name, function_args, allow_high_risk=False):
        from app.services.strains import strain_service

        pending_id = strain_service.create_pending_workflow_subculture(
            function_args["strain_id"],
            requester="agent_tool",
            session_id=function_args.get("session_id"),
            agent_run_id=function_args.get("agent_run_id"),
            source_message=function_args.get("source_message"),
        )
        return {
            "response_payload": {
                "action": "workflow_subculture",
                "status": "pending",
                "pending_id": pending_id,
                "strain_id": function_args["strain_id"],
                "require_confirmation": True,
            },
            "memory_text": "workflow pending",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="run subculture workflow for due strains",
        session_id="dynamic-replan-single-s1",
    )))

    assert response.agent_output["action"] == "workflow_subculture"
    assert response.agent_output["status"] == "pending"
    assert response.agent_output["strain_id"] == "Chlorella_01"
    assert _event_count("run-dynamic-replan-single-1", "agent_step_started") == 2
    replan = _event_payload("run-dynamic-replan-single-1", "agent_replan_decision_made")
    assert replan["replan_source"] == "dynamic_observation"
    assert replan["decision"]["decision_source"] == "dynamic_replan"
    assert replan["decision"]["route_kind"] == "workflow"


def test_dynamic_replan_multiple_due_strains_requires_clarification(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-dynamic-replan-multi-1")
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context(
        strains=[
            {
                "strain_id": "Chlorella_01",
                "name_cn": "chlorella",
                "name_en": "Chlorella vulgaris",
                "generation_number": 8,
                "days_since_last_subculture": 9,
            },
            {
                "strain_id": "Spirulina_02",
                "name_cn": "spirulina",
                "name_en": "Spirulina platensis",
                "generation_number": 4,
                "days_since_last_subculture": 10,
            },
        ]
    ))
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: QueryDecision(
        kind=RouteKind.LAB_QUERY,
        reason_code=ReasonCode.LAB_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="find due strains first",
        query_type="due_subculture",
    ))

    async def fail_tool_executor(*args, **kwargs):
        raise AssertionError("multiple dynamic workflow targets must not create pending requests")

    monkeypatch.setattr(chat_service, "execute_registered_tool", fail_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="run subculture workflow for due strains",
        session_id="dynamic-replan-multi-s1",
    )))

    assert response.agent_output["action"] == "dynamic_replan_clarification"
    assert response.agent_output["status"] == "needs_clarification"
    assert _event_count("run-dynamic-replan-multi-1", "agent_step_started") == 2
    assert _event_count("run-dynamic-replan-multi-1", "agent_replan_decision_made") == 1


def test_llm_replan_can_generate_clarification_without_tool_dispatch(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-llm-replan-clarify-1")
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: KnowledgeDecision(
        reason_code=ReasonCode.KNOWLEDGE_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="rag first",
        source_text="summarize the observation",
    ))

    def fake_rag_handler(session_id, conversation_history, message, agent_run_id=None):
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={"action": "rag_answer", "status": "success", "answer": {"conclusion": "ambiguous"}},
            natural_reply="ambiguous result",
        )

    def fake_llm_suggestions(state, observation):
        return LlmReplanSuggestion(
            assessment=LlmObservationAssessment(
                category="insufficient_evidence",
                explanation="The observation needs a target clarification.",
                confidence=0.8,
            ),
            candidates=(
                LlmReplanCandidate(
                    action_type="ask_clarification",
                    confidence=0.9,
                    reason="The observation does not identify one target.",
                    question="Which strain should I use?",
                ),
            ),
        )

    async def fail_tool_executor(*args, **kwargs):
        raise AssertionError("LLM clarification replanning must not call tools")

    monkeypatch.setattr(dispatcher, "handle_rag_intent", fake_rag_handler)
    monkeypatch.setattr(replanning, "collect_llm_replan_suggestions", fake_llm_suggestions)
    monkeypatch.setattr(chat_service, "execute_registered_tool", fail_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="summarize the observation",
        session_id="llm-replan-clarify-s1",
    )))

    assert response.agent_output["action"] == "dynamic_replan_clarification"
    assert response.agent_output["status"] == "needs_clarification"
    directive = _event_payload("run-llm-replan-clarify-1", "agent_replan_directive_created")
    assert directive["directive"]["llm_assessment"]["category"] == "insufficient_evidence"
    assert directive["directive"]["selected_candidate"]["source"] == "llm"


def test_llm_replan_workflow_candidate_still_creates_only_pending(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-llm-replan-workflow-1")
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context(
        strains=[
            {
                "strain_id": "Chlorella_01",
                "name_cn": "chlorella",
                "name_en": "Chlorella vulgaris",
                "generation_number": 8,
                "days_since_last_subculture": 9,
            }
        ]
    ))
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: QueryDecision(
        kind=RouteKind.LAB_QUERY,
        reason_code=ReasonCode.LAB_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="read due strains",
        query_type="due_subculture",
    ))

    def fake_llm_suggestions(state, observation):
        return LlmReplanSuggestion(
            assessment=LlmObservationAssessment(
                category="success",
                explanation="One due strain was found.",
                confidence=0.9,
            ),
            candidates=(
                LlmReplanCandidate(
                    action_type="propose_workflow_approval",
                    confidence=0.92,
                    reason="The observation has one due strain and approval can be proposed.",
                    target_ids=("Chlorella_01",),
                ),
            ),
        )

    async def fake_tool_executor(tool_name, function_args, allow_high_risk=False):
        from app.services.strains import strain_service

        pending_id = strain_service.create_pending_workflow_subculture(
            function_args["strain_id"],
            requester="agent_tool",
            session_id=function_args.get("session_id"),
            agent_run_id=function_args.get("agent_run_id"),
            source_message=function_args.get("source_message"),
        )
        return {
            "response_payload": {
                "action": "workflow_subculture",
                "status": "pending",
                "pending_id": pending_id,
                "strain_id": function_args["strain_id"],
                "require_confirmation": True,
            },
            "memory_text": "workflow pending",
        }

    monkeypatch.setattr(replanning, "collect_llm_replan_suggestions", fake_llm_suggestions)
    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="which strains are due now?",
        session_id="llm-replan-workflow-s1",
    )))

    assert response.agent_output["action"] == "workflow_subculture"
    assert response.agent_output["status"] == "pending"
    assert response.agent_output["pending_id"]
    policies = _event_payloads("run-llm-replan-workflow-1", "agent_policy_evaluated")
    assert policies[-1]["policy"]["category"] == "require_approval"
    directive = _event_payload("run-llm-replan-workflow-1", "agent_replan_directive_created")
    assert directive["directive"]["selected_candidate"]["source"] == "llm"
    assert directive["directive"]["selected_candidate"]["action_type"] == "propose_workflow_approval"


def test_llm_replan_cannot_trigger_workflow_from_rag_observation(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context(
        strains=[{"strain_id": "Chlorella_01", "name_cn": "chlorella", "name_en": "Chlorella vulgaris"}]
    ))
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-llm-replan-rag-block-1")
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: KnowledgeDecision(
        reason_code=ReasonCode.KNOWLEDGE_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="rag first",
        source_text="can I subculture based on docs",
    ))

    def fake_rag_handler(session_id, conversation_history, message, agent_run_id=None):
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={"action": "rag_answer", "status": "success", "answer": {"conclusion": "looks due"}},
            natural_reply="looks due",
        )

    def fake_llm_suggestions(state, observation):
        return LlmReplanSuggestion(
            assessment=LlmObservationAssessment(
                category="success",
                explanation="The text suggests a possible workflow.",
                confidence=0.9,
            ),
            candidates=(
                LlmReplanCandidate(
                    action_type="propose_workflow_approval",
                    confidence=0.95,
                    reason="LLM thinks workflow might be appropriate.",
                    target_ids=("Chlorella_01",),
                ),
            ),
        )

    async def fail_tool_executor(*args, **kwargs):
        raise AssertionError("RAG observation must not trigger workflow tools")

    monkeypatch.setattr(dispatcher, "handle_rag_intent", fake_rag_handler)
    monkeypatch.setattr(replanning, "collect_llm_replan_suggestions", fake_llm_suggestions)
    monkeypatch.setattr(chat_service, "execute_registered_tool", fail_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="can I subculture based on docs",
        session_id="llm-replan-rag-block-s1",
    )))

    assert response.agent_output["action"] == "rag_answer"
    assert _event_count("run-llm-replan-rag-block-1", "agent_step_started") == 1
    directive = _event_payload("run-llm-replan-rag-block-1", "agent_replan_directive_created")
    assert directive["directive"]["blocked_by"] == "rag_boundary"
    assert directive["directive"]["llm_candidates"][0]["blocked_by"] == "rag_boundary"


def test_dynamic_replan_existing_pending_returns_boundary_without_duplicate(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-dynamic-replan-existing-pending-1")
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context(
        strains=[
            {
                "strain_id": "Chlorella_01",
                "name_cn": "chlorella",
                "name_en": "Chlorella vulgaris",
                "generation_number": 8,
                "days_since_last_subculture": 9,
            }
        ],
        pending_actions=[
            {
                "id": 42,
                "action_type": "workflow_subculture",
                "status": "pending",
                "requester": "agent_tool",
                "payload": {"type": "workflow_subculture", "data": {"strain_id": "Chlorella_01"}},
            }
        ],
    ))
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: QueryDecision(
        kind=RouteKind.LAB_QUERY,
        reason_code=ReasonCode.LAB_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="find due strains first",
        query_type="due_subculture",
    ))

    async def fail_tool_executor(*args, **kwargs):
        raise AssertionError("existing pending boundary must not create a duplicate pending request")

    monkeypatch.setattr(chat_service, "execute_registered_tool", fail_tool_executor)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="run subculture workflow for due strains",
        session_id="dynamic-replan-existing-pending-s1",
    )))

    assert response.agent_output["action"] == "dynamic_replan_explain_result"
    assert response.agent_output["status"] == "success"
    assert "pending 42" in response.natural_reply
    assert _event_count("run-dynamic-replan-existing-pending-1", "agent_step_started") == 2
    directive = _event_payload("run-dynamic-replan-existing-pending-1", "agent_replan_directive_created")
    assert directive["directive"]["selected_candidate"]["metadata"]["policy_boundary"] == "existing_workflow_pending"
