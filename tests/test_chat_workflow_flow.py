import asyncio

from app.schemas.algae import ChatRequest
from app.services.chat import chat_service
from app.services.intent.routing_models import (
    EntityRef,
    ReasonCode,
    RiskLevel,
    WorkflowDecision,
)

from tests.conftest import context


def test_handle_chat_routes_explicit_workflow_to_pending(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service, "build_routing_decision", lambda routing_input: WorkflowDecision(
        reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
        risk_level=RiskLevel.HIGH,
        explanation="test explicit workflow",
        target=EntityRef("strain", "Chlorella_01", "Chlorella_01", "database_id", True),
    ))

    async def fail_llm_path(*args, **kwargs):
        raise AssertionError("LLM fallback should not handle explicit workflow intent")

    monkeypatch.setattr(chat_service, "_handle_llm_or_tool_path", fail_llm_path)
    from app.services.chat import workflow_intent_handler

    monkeypatch.setattr(
        workflow_intent_handler.strain_service,
        "create_pending_workflow_subculture",
        lambda strain_id, requester="LLM_workflow", session_id=None, agent_run_id=None, source_message=None: 51,
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
    called = {"count": 0}

    async def fake_execute_registered_tool(tool_name, function_args, allow_high_risk=False):
        called["count"] += 1
        assert tool_name == "trigger_subculture_workflow"
        assert allow_high_risk is True
        return {
            "response_payload": {
                "action": "workflow_subculture",
                "status": "pending",
                "strain_id": function_args["strain_id"],
                "pending_id": 51,
                "require_confirmation": True,
            },
            "memory_text": "workflow pending created",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_execute_registered_tool)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="run workflow",
        session_id="chat-workflow-s1",
    )))

    assert called["count"] == 1
    assert response.agent_output["action"] == "workflow_subculture"
    assert response.agent_output["status"] == "pending"
    assert response.agent_output["pending_id"] == 51
