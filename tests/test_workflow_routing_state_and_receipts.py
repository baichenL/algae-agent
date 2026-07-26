import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.core import database
from app.core.db import workflow_runs
from app.schemas.algae import ChatRequest, ChatResponse
from app.services.chat import chat_service, response_builder, workflow_request_state
from app.services.chat.operational_claim_guard import guard_chat_reply
from app.services.intent.input_normalizer import normalize_input
from app.services.intent.intent_router import build_routing_decision
from app.services.intent.routing_models import (
    ChatDecision,
    ReasonCode,
    RiskLevel,
    RouteKind,
    RoutingInput,
    SpeechAct,
    WorkflowAuditDecision,
    WorkflowDecision,
)
from app.services.strains import strain_service
from app.services.workflows import workflow_execution_service


def _context():
    return SimpleNamespace(
        strains=[
            {
                "strain_id": "Chlamydomonas_137AH",
                "name_cn": "莱茵衣藻",
                "name_en": "Chlamydomonas reinhardtii",
                "generation_number": 3,
                "days_since_last_subculture": 7,
            }
        ],
        pending_actions=[],
        target_pending_actions=[],
        to_prompt_facts=lambda: "workflow test context",
    )


def _decision(message, *, active_workflow=None):
    return build_routing_decision(RoutingInput(
        message=normalize_input(message),
        session_id="workflow-route",
        context_snapshot=_context(),
        active_workflow_request=active_workflow,
    ))


def test_db_first_entity_resolution_accepts_id_adjacent_to_chinese():
    decision = _decision("为Chlamydomonas_137AH执行传代")

    assert isinstance(decision, WorkflowDecision)
    assert decision.target.canonical_id == "Chlamydomonas_137AH"


def test_workflow_diagnostic_outranks_command_and_rag():
    decision = _decision("为什么执行了传代但是数据库里没有变化")

    assert isinstance(decision, WorkflowAuditDecision)
    assert decision.kind == RouteKind.WORKFLOW_AUDIT
    assert decision.speech_act == SpeechAct.DIAGNOSTIC


def test_chat_confirmation_is_audit_only_not_execution():
    decision = _decision("确认执行")

    assert isinstance(decision, WorkflowAuditDecision)
    assert decision.speech_act == SpeechAct.CONFIRM


def test_active_workflow_accepts_bare_strain_id_as_slot():
    decision = _decision(
        "Chlamydomonas_137AH",
        active_workflow={"active": True, "state": "collecting_target"},
    )

    assert isinstance(decision, WorkflowDecision)
    assert decision.speech_act == SpeechAct.SLOT_VALUE
    assert decision.target.canonical_id == "Chlamydomonas_137AH"


def test_workflow_collection_creates_real_pending_on_second_turn(
    tmp_path,
    monkeypatch,
    isolated_sqlite_db,
):
    monkeypatch.setattr(workflow_request_state, "STATE_DIR", str(tmp_path / "workflow-state"))
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)
    database.add_algae_strain(
        "Chlamydomonas_137AH",
        "莱茵衣藻",
        "Chlamydomonas reinhardtii",
        generation_number=3,
        days_since_last_subculture=7,
    )

    first = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="执行传代",
        session_id="workflow-collect",
    )))
    assert first.agent_output["action"] == "workflow_request_clarification"
    assert database.list_pending_actions() == []
    assert workflow_request_state.get_workflow_request_state("workflow-collect") is not None

    second = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="Chlamydomonas_137AH",
        session_id="workflow-collect",
    )))
    assert second.agent_output["action"] == "workflow_subculture"
    assert second.agent_output["pending_id"]
    assert second.agent_output["requires_approval"] is True
    assert second.agent_output["protocol_summary"]["target_strain_id"] == "Chlamydomonas_137AH"
    assert second.agent_output["validation_summary"]["valid"] is True
    assert second.agent_output["simulation_summary"]["success"] is True
    assert second.agent_output["run_preview"]
    assert len(database.list_pending_actions()) == 1
    assert workflow_request_state.get_workflow_request_state("workflow-collect") is None


def test_multi_target_workflow_clarification_keeps_pending_creation_state(
    tmp_path,
    monkeypatch,
    isolated_sqlite_db,
):
    # 中文场景回归：中文品系名命中多个 strain 时，第二轮裸 strain_id
    # 必须继续原 workflow 补槽，不能退回普通聊天。
    monkeypatch.setattr(workflow_request_state, "STATE_DIR", str(tmp_path / "workflow-state"))
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)
    database.add_algae_strain(
        "Chlamydomonas_01",
        "莱茵衣藻",
        "Chlamydomonas reinhardtii",
        generation_number=2,
        days_since_last_subculture=8,
    )
    database.add_algae_strain(
        "Chlamydomonas_137AH",
        "莱茵衣藻",
        "Chlamydomonas reinhardtii",
        generation_number=3,
        days_since_last_subculture=7,
    )

    first = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="为莱茵衣藻执行传代",
        session_id="workflow-multi-target",
    )))
    assert first.agent_output["action"] == "routing_clarification"
    assert first.agent_output["reason_code"] == "multiple_targets_conflict"
    assert database.list_pending_actions() == []
    assert workflow_request_state.get_workflow_request_state("workflow-multi-target") is not None

    second = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="Chlamydomonas_137AH",
        session_id="workflow-multi-target",
    )))
    assert second.agent_output["action"] == "workflow_subculture"
    assert second.agent_output["pending_id"]
    assert len(database.list_pending_actions(status="pending")) == 1
    assert workflow_request_state.get_workflow_request_state("workflow-multi-target") is None


def test_unrelated_chat_keeps_workflow_collection_state(tmp_path, monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(workflow_request_state, "STATE_DIR", str(tmp_path / "workflow-state"))
    state = workflow_request_state.build_workflow_request_state("workflow-chat", "run-1", "执行传代")
    workflow_request_state.save_workflow_request_state("workflow-chat", state)
    decision = _decision("你好", active_workflow=state)

    assert decision.kind == RouteKind.CHAT
    assert workflow_request_state.get_workflow_request_state("workflow-chat") is not None


def test_chat_operational_claim_is_blocked(monkeypatch):
    events = []
    from app.services.chat import operational_claim_guard
    monkeypatch.setattr(operational_claim_guard, "append_decision_event", events.append)

    reply, blocked = guard_chat_reply(
        "正在调用硬件流水线。[工具调用已触发]",
        session_id="claim-test",
        agent_run_id="run-claim",
    )

    assert blocked is True
    assert "没有检测到真实 Pending" in reply
    assert events[0]["event"] == "unsupported_operational_claim_blocked"


def test_capability_question_uses_local_reply_without_llm(monkeypatch):
    from app.services.chat import llm_chat_handler

    def fail_if_called(*args, **kwargs):
        raise AssertionError("capability question should not call external LLM")

    monkeypatch.setattr(llm_chat_handler.client.chat.completions, "create", fail_if_called)
    saved = []
    monkeypatch.setattr(llm_chat_handler, "save_session_memory", lambda session_id, history: saved.append((session_id, history)))
    decision = ChatDecision(
        reason_code=ReasonCode.DEFAULT_CHAT,
        risk_level=RiskLevel.NONE,
        explanation="capability question",
        source_text="你能完成哪些工作？",
    )

    response = asyncio.run(llm_chat_handler.handle_llm_or_tool_path(
        "capability-session",
        [{"role": "system", "content": "test"}],
        decision,
        SimpleNamespace(to_prompt_facts=lambda: "context"),
        "run-capability",
    ))

    assert response.agent_output["action"] == "capability_summary"
    assert response.agent_output["operational_claim_blocked"] is False
    assert "查询实验室状态" in response.natural_reply
    assert saved[0][0] == "capability-session"


def test_assistant_history_endpoint_hides_system_prompt(tmp_path, monkeypatch):
    from app.api import v2
    from app.core import config

    monkeypatch.setenv("ALGAE_AUTH_MODE", "test")
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    config.save_memory("react-assistant", [
        {"role": "system", "content": "secret system prompt"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我在。"},
    ])
    app = FastAPI()
    app.include_router(v2.router)

    response = TestClient(app).get("/api/v2/assistant/history?session_id=react-assistant")

    assert response.status_code == 200
    assert response.json()["messages"] == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，我在。"},
    ]


def test_approved_simulation_creates_run_without_mutating_strain(
    isolated_sqlite_db,
    monkeypatch,
):
    before = database.get_algae_status("Chlorella_01")
    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")

    async def fake_simulation(strain_id, execution_mode=None):
        return {
            "status": "success",
            "strain_id": strain_id,
            "execution_mode": "simulation",
            "physical_execution": False,
            "persisted": False,
            "current_generation": before["generation_number"] + 1,
            "days_counter": 0,
            "inoculation_time": "2026-06-23 12:00:00",
            "execution_logs": ["simulation complete"],
        }

    monkeypatch.setattr(workflow_execution_service, "run_force_subculture_workflow", fake_simulation)
    result = asyncio.run(strain_service.approve_pending_action_async(pending_id))

    assert result["status"] == "success"
    assert result["physical_execution"] is False
    assert result["persisted"] is False
    assert database.get_algae_status("Chlorella_01")["generation_number"] == before["generation_number"]
    run = workflow_runs.get_workflow_run_by_pending(pending_id)
    assert run["status"] == "succeeded"
    assert run["tool_result"]


def test_workflow_approval_api_returns_202_and_run_id(
    isolated_sqlite_db,
    monkeypatch,
):
    from app.api import strain as strain_api
    from app.services import approval_execution_service
    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")
    scheduled = []

    async def fake_background_execution(run_id):
        scheduled.append(run_id)
        return {"status": "success", "run_id": run_id}

    monkeypatch.setattr(
        approval_execution_service,
        "execute_workflow_run",
        fake_background_execution,
    )
    app = FastAPI()
    app.include_router(strain_api.router, prefix="/api/v1")
    response = TestClient(app).post(
        "/api/v1/strain/confirm",
        json={"pending_id": pending_id, "approve": True},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["pending_id"] == pending_id
    assert body["run_id"]
    assert body["executed"] is False
    assert scheduled == [body["run_id"]]


def test_memory_loader_replaces_stale_system_prompt(tmp_path, monkeypatch):
    from app.core import config
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    history_path = tmp_path / "history_stale.json"
    history_path.write_text(json.dumps([
        {"role": "system", "content": "必须调用 trigger_subculture_workflow"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好"},
    ], ensure_ascii=False), encoding="utf-8")

    loaded = config.load_memory("stale")

    assert "普通 Chat 没有工具权限" in loaded[0]["content"]
    assert all("trigger_subculture_workflow" not in item["content"] for item in loaded)
