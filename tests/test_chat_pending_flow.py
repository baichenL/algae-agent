import asyncio

from app.schemas.algae import ChatRequest, ChatResponse
from app.services.chat import chat_service
from app.services.chat import pending_form_state as pfs

from tests.conftest import context


def _patch_chat_basics(monkeypatch, tmp_path, ctx=None):
    monkeypatch.setattr(pfs, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: ctx or context())


def test_first_add_turn_requires_more_info_and_does_not_call_tool(tmp_path, monkeypatch, isolated_sqlite_db):
    _patch_chat_basics(monkeypatch, tmp_path)
    called = {"count": 0}

    async def fake_execute(tool_name, function_args, allow_high_risk=False):
        called["count"] += 1
        return {}

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_execute)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="增加螺旋藻", session_id="s1")))

    assert response.agent_output["action"] == "require_more_info"
    assert response.agent_output["require_more_info"] is True
    assert called["count"] == 0
    assert pfs.get_pending_form_state("s1") is not None


def test_second_add_turn_creates_pending_and_clears_state(tmp_path, monkeypatch, isolated_sqlite_db):
    _patch_chat_basics(monkeypatch, tmp_path)
    asyncio.run(chat_service.handle_chat(ChatRequest(message="增加螺旋藻", session_id="s2")))
    called = {"args": None}

    async def fake_execute(tool_name, function_args, allow_high_risk=False):
        called["args"] = function_args
        return {
            "response_payload": {
                "action": "add_strain",
                "status": "pending",
                "pending_id": 7,
                "require_confirmation": True,
            },
            "memory_text": "Add strain request created, pending_id=7",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_execute)

    response = asyncio.run(chat_service.handle_chat(
        ChatRequest(message="Spirulina_02 | 螺旋藻 (Spirulina platensis)", session_id="s2")
    ))

    assert response.agent_output["pending_id"] == 7
    assert called["args"]["strain_id"] == "Spirulina_02"
    assert pfs.get_pending_form_state("s2") is None


def test_active_state_pending_query_keeps_state(tmp_path, monkeypatch, isolated_sqlite_db):
    _patch_chat_basics(monkeypatch, tmp_path)
    state = {
        "active": True,
        "operation": "add",
        "tool_name": "add_algae_strain",
        "collected_fields": {"name_cn": "螺旋藻"},
        "missing_fields": ["strain_id", "name_en"],
        "candidates": [],
        "source_message": "增加螺旋藻",
    }
    pfs.save_pending_form_state("s3", state)
    monkeypatch.setattr(
        chat_service,
        "handle_query_status_intent",
        lambda decision, context_snapshot: {
            "agent_output": {"action": "list_pending", "pending": []},
            "natural_reply": "暂无待审批请求",
        },
    )

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="待审批列表", session_id="s3")))

    assert response.agent_output["action"] == "list_pending"
    assert pfs.get_pending_form_state("s3") is not None


def test_unrelated_chat_does_not_become_pending_form_continuation(tmp_path, monkeypatch, isolated_sqlite_db):
    _patch_chat_basics(monkeypatch, tmp_path)
    asyncio.run(chat_service.handle_chat(ChatRequest(message="增加螺旋藻", session_id="chat-during-form")))

    async def fake_llm(session_id, conversation_history, decision, context_snapshot, agent_run_id):
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={"action": "chat", "content": "我是微藻实验助手。"},
            natural_reply="我是微藻实验助手。",
        )

    monkeypatch.setattr(chat_service, "_handle_llm_or_tool_path", fake_llm)
    response = asyncio.run(chat_service.handle_chat(ChatRequest(
        message="你好，简单介绍一下你自己",
        session_id="chat-during-form",
    )))

    assert response.agent_output["action"] == "chat"
    assert pfs.get_pending_form_state("chat-during-form") is not None


def test_cancel_clears_pending_form_state(tmp_path, monkeypatch, isolated_sqlite_db):
    _patch_chat_basics(monkeypatch, tmp_path)
    pfs.save_pending_form_state("s4", {
        "active": True,
        "operation": "add",
        "tool_name": "add_algae_strain",
        "collected_fields": {"name_cn": "螺旋藻"},
        "missing_fields": ["strain_id", "name_en"],
        "candidates": [],
        "source_message": "增加螺旋藻",
    })

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="取消", session_id="s4")))

    assert response.agent_output["status"] == "cancelled"
    assert pfs.get_pending_form_state("s4") is None


def test_complete_fields_call_tool_but_missing_pending_id_is_not_success(tmp_path, monkeypatch, isolated_sqlite_db):
    _patch_chat_basics(monkeypatch, tmp_path)

    async def fake_execute(tool_name, function_args, allow_high_risk=False):
        return {
            "response_payload": {
                "action": "add_strain",
                "status": "pending",
                "require_confirmation": True,
            },
            "memory_text": "pending missing",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_execute)

    response = asyncio.run(chat_service.handle_chat(
        ChatRequest(message="增加 Spirulina_02 | 螺旋藻 (Spirulina platensis)", session_id="s5")
    ))

    assert "请求未创建成功" in response.natural_reply
    assert not response.agent_output.get("pending_id")


def test_active_state_new_write_intent_returns_conflict(tmp_path, monkeypatch, isolated_sqlite_db):
    _patch_chat_basics(monkeypatch, tmp_path)
    pfs.save_pending_form_state("s6", {
        "active": True,
        "operation": "add",
        "tool_name": "add_algae_strain",
        "collected_fields": {"name_cn": "螺旋藻"},
        "missing_fields": ["strain_id", "name_en"],
        "candidates": [],
        "source_message": "增加螺旋藻",
    })

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="删除小球藻", session_id="s6")))

    assert response.agent_output["action"] == "pending_form_conflict"
    assert pfs.get_pending_form_state("s6") is not None
