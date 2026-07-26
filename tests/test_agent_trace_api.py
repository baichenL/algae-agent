import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router as api_router
from app.schemas.algae import ChatRequest
from app.services.chat import chat_service, response_builder
from app.services.intent.routing_models import EntityRef, QueryDecision, ReasonCode, RiskLevel, RouteKind, WorkflowDecision
from tests.conftest import context


def _client():
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def test_agent_trace_api_returns_steps_and_raw_events(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "trace-run-1")

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="pending", session_id="trace-s1")))

    assert response.agent_output["action"] == "list_pending"
    result = _client().get("/api/v1/agent/runs/trace-run-1")

    assert result.status_code == 200
    body = result.json()
    assert body["run"]["id"] == "trace-run-1"
    assert body["trace"]["steps"][0]["decision"]["route_kind"] == "pending_query"
    assert body["trace"]["steps"][0]["plan_step"]["route_kind"] == "pending_query"
    assert body["trace"]["steps"][0]["policy_verdict"]["category"] == "allow_read_only"
    assert body["trace"]["steps"][0]["observation"]["action"] == "list_pending"
    assert body["trace"]["steps"][0]["observation_assessment"]["reason"] == "observation_success"
    assert body["trace"]["steps"][0]["replan_directive"]["reason"] == "no_loop_plan"
    assert body["trace"]["context"]["fact_context"]["authority"] == "database_current_state"
    assert body["trace"]["status_bars"]
    assert body["trace"]["status_bars"][-1]["status_bar"]["step"] == "1/3"
    assert body["trace"]["plan"]["mode"] == "single_step"
    assert body["raw_events"]


def test_agent_trace_api_returns_404_for_missing_run(isolated_sqlite_db):
    result = _client().get("/api/v1/agent/runs/missing-run")

    assert result.status_code == 404


def test_agent_trace_api_shows_workflow_requires_approval(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "trace-workflow-run-1")
    target = EntityRef("strain", "Chlorella_01", "Chlorella_01", "database_id", True)
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: WorkflowDecision(
        reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
        risk_level=RiskLevel.HIGH,
        explanation="workflow trace",
        target=target,
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
            "pending_id": pending_id,
            "response_payload": {
                "action": "workflow_subculture",
                "status": "pending",
                "pending_id": pending_id,
                "require_confirmation": True,
            },
            "memory_text": "workflow pending",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_tool_executor)

    asyncio.run(chat_service.handle_chat(ChatRequest(message="run workflow", session_id="trace-wf-s1")))
    result = _client().get("/api/v1/agent/runs/trace-workflow-run-1")

    assert result.status_code == 200
    step = result.json()["trace"]["steps"][0]
    assert step["decision"]["route_kind"] == "workflow"
    assert step["policy_verdict"]["category"] == "require_approval"
    assert step["policy_verdict"]["requires_approval"] is True
    assert step["observation"]["pending_id"]


def test_agent_trace_api_shows_dynamic_replan_source(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "trace-dynamic-replan-run-1")
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
            "pending_id": pending_id,
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

    asyncio.run(chat_service.handle_chat(ChatRequest(
        message="run subculture workflow for due strains",
        session_id="trace-dynamic-s1",
    )))
    result = _client().get("/api/v1/agent/runs/trace-dynamic-replan-run-1")

    assert result.status_code == 200
    steps = result.json()["trace"]["steps"]
    assert len(steps) == 2
    assert steps[0]["replan_source"] == "dynamic_observation"
    assert steps[0]["replan_decision"]["decision_source"] == "dynamic_replan"
    assert steps[1]["decision"]["route_kind"] == "workflow"
    assert steps[1]["observation"]["pending_id"]


def test_agent_trace_api_keeps_envelope_for_early_failure(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "trace-context-failure-run-1")

    def fail_context(session_id):
        raise RuntimeError("eval context failure")

    monkeypatch.setattr(chat_service, "build_context_snapshot", fail_context)

    try:
        asyncio.run(chat_service.handle_chat(ChatRequest(message="pending", session_id="trace-fail-s1")))
    except RuntimeError:
        pass

    result = _client().get("/api/v1/agent/runs/trace-context-failure-run-1")

    assert result.status_code == 200
    trace = result.json()["trace"]
    assert trace["final_status"] == "failed"
    assert trace["failure_stage"] == "build_task_spec"
    assert trace["failure_layer"] == "chat_service"
    assert trace["error_event_id"]
    assert trace["errors"][0]["operation"] == "build_task_spec"
    assert [item["event_type"] for item in trace["lifecycle_events"]] == ["run_started", "run_failed"]


def test_agent_run_list_api_filters_by_session(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "trace-list-run-1")

    asyncio.run(chat_service.handle_chat(ChatRequest(message="pending", session_id="trace-list-s1")))
    result = _client().get("/api/v1/agent/runs?session_id=trace-list-s1")

    assert result.status_code == 200
    runs = result.json()["runs"]
    assert [item["id"] for item in runs] == ["trace-list-run-1"]
