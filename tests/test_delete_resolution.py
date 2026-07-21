import asyncio

from app.schemas.algae import ChatRequest
from app.services.chat import chat_service
from app.services.chat import pending_form_state as pfs

from tests.conftest import context


def test_delete_requires_unique_strain_id(tmp_path, monkeypatch, isolated_sqlite_db):
    ctx = context(strains=[
        {
            "strain_id": "Chlamydomonas_01",
            "name_cn": "莱茵衣藻",
            "name_en": "Chlamydomonas reinhardtii",
        },
        {
            "strain_id": "Chlamydomonas_137AH",
            "name_cn": "莱茵衣藻",
            "name_en": "Chlamydomonas reinhardtii mutant",
        },
    ])
    monkeypatch.setattr(pfs, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: ctx)
    called = {"count": 0}

    async def fake_execute(tool_name, function_args, allow_high_risk=False):
        called["count"] += 1
        return {}

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_execute)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="删除莱茵衣藻", session_id="delete-s1")))

    assert response.agent_output["action"] == "require_more_info"
    assert called["count"] == 0
    assert "Chlamydomonas_01" in response.natural_reply
    assert "Chlamydomonas_137AH" in response.natural_reply


def test_delete_unique_strain_id_can_create_pending(tmp_path, monkeypatch, isolated_sqlite_db):
    ctx = context(strains=[
        {
            "strain_id": "Chlamydomonas_01",
            "name_cn": "莱茵衣藻",
            "name_en": "Chlamydomonas reinhardtii",
        }
    ])
    monkeypatch.setattr(pfs, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: ctx)

    async def fake_execute(tool_name, function_args, allow_high_risk=False):
        return {
            "response_payload": {
                "action": "delete_strain",
                "status": "pending",
                "pending_id": 9,
                "require_confirmation": True,
            },
            "memory_text": "Delete strain request created, pending_id=9",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_execute)

    response = asyncio.run(chat_service.handle_chat(ChatRequest(message="删除莱茵衣藻", session_id="delete-s2")))

    assert response.agent_output["pending_id"] == 9
    assert response.agent_output["action"] == "delete_strain"
