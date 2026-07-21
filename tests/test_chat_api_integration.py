from app.core import database
from app.services.chat import pending_form_state as pfs


def _post_chat(client, message, session_id):
    return client.post(
        "/api/v1/chat",
        json={"message": message, "session_id": session_id},
    )


def test_chat_api_first_add_turn_requires_more_info(
    api_client,
    isolated_pending_form_state,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    session_id = "api-add-first"

    response = _post_chat(api_client, "增加螺旋藻", session_id)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["session_id"] == session_id
    assert body["agent_output"]["action"] == "require_more_info"
    assert body["agent_output"]["require_more_info"] is True
    assert pfs.get_pending_form_state(session_id) is not None
    assert database.list_pending_actions() == []


def test_chat_api_second_turn_creates_real_pending_action(
    api_client,
    isolated_pending_form_state,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    session_id = "api-add-second"

    first = _post_chat(api_client, "增加螺旋藻", session_id)
    assert first.status_code == 200

    second = _post_chat(
        api_client,
        "Spirulina_02 | 螺旋藻 (Spirulina platensis)",
        session_id,
    )

    assert second.status_code == 200
    body = second.json()
    assert body["agent_output"]["action"] == "add_strain"
    assert body["agent_output"]["pending_id"]
    assert pfs.get_pending_form_state(session_id) is None

    pending = database.list_pending_actions()
    assert len(pending) == 1
    assert pending[0]["action_type"] == "add_strain"
    assert pending[0]["payload"]["data"]["strain_id"] == "Spirulina_02"
    assert pending[0]["payload"]["data"]["name_cn"] == "螺旋藻"
    assert pending[0]["payload"]["data"]["name_en"] == "Spirulina platensis"


def test_chat_api_pending_query_keeps_active_state(
    api_client,
    isolated_pending_form_state,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    session_id = "api-query-keeps-state"

    first = _post_chat(api_client, "增加螺旋藻", session_id)
    assert first.status_code == 200
    before_state = pfs.get_pending_form_state(session_id)
    assert before_state is not None

    query = _post_chat(api_client, "待审批列表", session_id)

    assert query.status_code == 200
    body = query.json()
    assert body["agent_output"]["action"] == "list_pending"
    after_state = pfs.get_pending_form_state(session_id)
    assert after_state is not None
    assert after_state["operation"] == before_state["operation"]
    assert after_state["missing_fields"] == before_state["missing_fields"]


def test_chat_api_cancel_clears_pending_form_state(
    api_client,
    isolated_pending_form_state,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    session_id = "api-cancel-state"

    first = _post_chat(api_client, "增加螺旋藻", session_id)
    assert first.status_code == 200
    assert pfs.get_pending_form_state(session_id) is not None

    cancel = _post_chat(api_client, "取消", session_id)

    assert cancel.status_code == 200
    body = cancel.json()
    assert body["agent_output"]["status"] == "cancelled"
    assert pfs.get_pending_form_state(session_id) is None


def test_chat_api_tool_missing_pending_id_does_not_claim_success(
    api_client,
    isolated_pending_form_state,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
    monkeypatch,
):
    from app.services.chat import chat_service

    session_id = "api-missing-pending-id"

    async def fake_execute_registered_tool(tool_name, function_args, allow_high_risk=False):
        return {
            "response_payload": {
                "action": "add_strain",
                "status": "pending",
                "require_confirmation": True,
            },
            "memory_text": "pending missing",
        }

    monkeypatch.setattr(chat_service, "execute_registered_tool", fake_execute_registered_tool)

    response = _post_chat(
        api_client,
        "增加 Spirulina_02 | 螺旋藻 (Spirulina platensis)",
        session_id,
    )

    assert response.status_code == 200
    body = response.json()
    assert not body["agent_output"].get("pending_id")
    assert "已创建待确认请求" not in body["natural_reply"]
    assert "请求未创建成功" in body["natural_reply"]
    assert database.list_pending_actions() == []
