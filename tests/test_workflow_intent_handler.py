import asyncio

from app.services.chat import response_builder
from app.services.chat.workflow_intent_handler import handle_workflow_intent
from app.services.intent.routing_models import (
    EntityRef,
    ReasonCode,
    RiskLevel,
    WorkflowDecision,
)


def _history():
    return [{"role": "system", "content": "test"}]


def _decision(strain_id=None):
    target = None
    if strain_id:
        target = EntityRef("strain", strain_id, strain_id, "database_id", True)
    return WorkflowDecision(
        reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
        risk_level=RiskLevel.HIGH,
        explanation="test workflow",
        target=target,
    )


def test_workflow_decision_requires_strain_id(monkeypatch, tmp_path, isolated_sqlite_db):
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    from app.services.chat import workflow_request_state
    monkeypatch.setattr(workflow_request_state, "STATE_DIR", str(tmp_path))

    response = asyncio.run(handle_workflow_intent(
        "workflow-s2",
        _history(),
        _decision(),
    ))

    assert response.agent_output["action"] == "workflow_request_clarification"


def test_workflow_decision_creates_pending_with_strain_id(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    from app.services.chat import workflow_intent_handler

    events = []
    monkeypatch.setattr(workflow_intent_handler, "append_decision_event", events.append)
    monkeypatch.setattr(
        workflow_intent_handler.strain_service,
        "create_pending_workflow_subculture",
        lambda strain_id, requester="LLM_workflow", session_id=None, agent_run_id=None, source_message=None: 42,
    )
    monkeypatch.setattr(
        workflow_intent_handler.strain_service,
        "get_pending_action",
        lambda pending_id: {
            "id": pending_id,
            "status": "pending",
            "payload": {
                "type": "workflow_subculture",
                "data": {"strain_id": "Chlorella_01"},
            },
        },
    )

    response = asyncio.run(handle_workflow_intent(
        "workflow-s3",
        _history(),
        _decision("Chlorella_01"),
        "run-123",
        "run workflow",
    ))

    assert response.agent_output["action"] == "workflow_subculture"
    assert response.agent_output["status"] == "pending"
    assert response.agent_output["strain_id"] == "Chlorella_01"
    assert response.agent_output["pending_id"] == 42
    assert events[0]["event"] == "workflow_pending_created"
    assert events[0]["agent_run_id"] == "run-123"
