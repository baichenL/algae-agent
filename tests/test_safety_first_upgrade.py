import asyncio
from types import SimpleNamespace

from app.core import config, database
from app.schemas.algae import ChatRequest
from app.services.chat import chat_service
from app.services.chat.workflow_intent_handler import handle_workflow_intent
from app.services.context.context_builder import build_context_snapshot
from app.services.intent.routing_models import (
    EntityRef,
    ReasonCode,
    RiskLevel,
    WorkflowDecision,
)
from app.services.strains import strain_service


def test_default_system_prompt_is_db_first_and_safety_first():
    prompt = config.DEFAULT_SYSTEM_PROMPT

    assert "安全优先" in prompt
    assert "事实优先" in prompt
    assert "pending action" in prompt
    assert "session history" in prompt


def test_context_snapshot_includes_target_strain_and_target_pending(isolated_sqlite_db, monkeypatch):
    monkeypatch.setattr(strain_service, "append_decision_event", lambda event: None)
    pid = strain_service.create_pending_update_strain({
        "strain_id": "Chlorella_01",
        "generation_number": 15,
    })

    snapshot = build_context_snapshot("snapshot-target", strain_id="Chlorella_01")

    assert snapshot.target_strain["strain_id"] == "Chlorella_01"
    assert [item["id"] for item in snapshot.target_pending_actions] == [pid]
    assert snapshot.to_dict()["target_pending_actions"][0]["id"] == pid
    assert "target_strain" in snapshot.to_prompt_facts()


def test_workflow_intent_creates_pending_without_direct_tool_executor(
    isolated_sqlite_db,
    monkeypatch,
):
    from app.services.chat import response_builder
    from app.services.chat import workflow_intent_handler

    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    events = []
    monkeypatch.setattr(workflow_intent_handler, "append_decision_event", events.append)
    monkeypatch.setattr(strain_service, "append_decision_event", lambda event: events.append(event))
    called = {"count": 0}

    async def fake_tool_executor(tool_name, function_args, allow_high_risk=False):
        called["count"] += 1
        return {}

    decision = WorkflowDecision(
        reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
        risk_level=RiskLevel.HIGH,
        explanation="test workflow",
        target=EntityRef("strain", "Chlorella_01", "Chlorella_01", "database_id", True),
    )

    response = asyncio.run(handle_workflow_intent(
        "workflow-session",
        [{"role": "system", "content": "test"}],
        decision,
        "run-123",
        "请执行 Chlorella_01 传代",
    ))

    assert called["count"] == 0
    assert response.agent_output["action"] == "workflow_subculture"
    assert response.agent_output["status"] == "pending"
    assert response.agent_output["risk_level"] == "high"

    pending = database.list_pending_actions()
    assert len(pending) == 1
    assert pending[0]["action_type"] == "workflow_subculture"
    assert pending[0]["risk_level"] == "high"
    assert pending[0]["payload"]["data"]["agent_run_id"] == "run-123"
    assert any(event.get("event") == "workflow_pending_created" for event in events)


def test_workflow_pending_approval_rechecks_strain_before_execution(
    isolated_sqlite_db,
    monkeypatch,
):
    events = []
    monkeypatch.setattr(strain_service, "append_decision_event", events.append)
    from app.services.workflows import workflow_approval_service
    monkeypatch.setattr(workflow_approval_service, "append_decision_event", events.append)
    pending_id = strain_service.create_pending_workflow_subculture(
        "Chlorella_01",
        session_id="workflow-session",
        agent_run_id="run-456",
    )
    database.delete_algae_strain("Chlorella_01")

    result = asyncio.run(strain_service.approve_pending_action_async(pending_id))

    assert result["status"] == "error"
    assert result["reason"] == "strain_not_found"
    assert database.get_pending_action(pending_id)["status"] == "pending"
    assert any(event.get("event") == "pending_review_failed" for event in events)


def test_chat_workflow_path_creates_pending_and_logs_same_agent_run(
    monkeypatch,
    isolated_sqlite_db,
):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-fixed")
    events = []
    monkeypatch.setattr(chat_service, "append_decision_event", events.append)
    monkeypatch.setattr(strain_service, "append_decision_event", events.append)
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: WorkflowDecision(
        reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
        risk_level=RiskLevel.HIGH,
        explanation="test explicit workflow",
        target=EntityRef("strain", "Chlorella_01", "Chlorella_01", "database_id", True),
    ))

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="run workflow",
        session_id="chat-workflow-pending",
    )))

    assert response.agent_output["action"] == "workflow_subculture"
    assert response.agent_output["status"] == "pending"
    assert {event.get("agent_run_id") for event in events if event.get("event") in {"chat_intent_routed", "pending_created"}} == {"run-fixed"}
