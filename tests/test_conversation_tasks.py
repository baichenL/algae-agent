from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router
from app.core.db import assistant_conversations, conversation_tasks


def test_task_state_enforces_one_active_state_change_and_cas(isolated_sqlite_db):
    conversation = assistant_conversations.create_conversation(
        "pytest-approver",
        workspace_id="shared",
        title="task test",
    )
    first = conversation_tasks.create_task(
        conversation_id=conversation["id"],
        owner="pytest-approver",
        workspace_id="shared",
        goal_text="传代",
        task_type="subculture",
        status="collecting",
        state_changing=True,
        missing_slots=["strain_id"],
    )
    with pytest.raises(conversation_tasks.ActiveTaskConflict):
        conversation_tasks.create_task(
            conversation_id=conversation["id"],
            owner="pytest-approver",
            workspace_id="shared",
            goal_text="删除品系",
            task_type="strain_delete",
            status="collecting",
            state_changing=True,
        )

    ready = conversation_tasks.update_task(
        first["id"],
        expected_version=first["version"],
        status="ready",
        collected_slots={"strain_id": "Chlorella_01"},
        missing_slots=[],
    )
    assert ready["version"] == first["version"] + 1
    with pytest.raises(conversation_tasks.TaskVersionConflict):
        conversation_tasks.update_task(first["id"], expected_version=first["version"], status="cancelled")


def test_task_api_is_persistent_and_cancel_is_idempotent(isolated_sqlite_db):
    conversation = assistant_conversations.create_conversation(
        "pytest-approver",
        workspace_id="shared",
        title="api task test",
    )
    task = conversation_tasks.create_task(
        conversation_id=conversation["id"],
        owner="pytest-approver",
        workspace_id="shared",
        goal_text="为 Chlorella_01 创建传代请求",
        task_type="subculture",
        status="waiting_approval",
        state_changing=True,
        pending_id=123,
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    listed = client.get(f"/api/v2/assistant/conversations/{conversation['id']}/tasks?scope=active")
    assert listed.status_code == 200
    assert listed.json()["tasks"][0]["id"] == task["id"]

    detail = client.get(f"/api/v2/assistant/tasks/{task['id']}")
    assert detail.status_code == 200
    assert detail.json()["events"][0]["event_type"] == "task_created"

    headers = {"X-CSRF-Token": "test-csrf"}
    cancelled = client.post(f"/api/v2/assistant/tasks/{task['id']}/cancel", headers=headers)
    assert cancelled.status_code == 200
    assert cancelled.json()["task"]["status"] == "cancelled"
    repeated = client.post(f"/api/v2/assistant/tasks/{task['id']}/cancel", headers=headers)
    assert repeated.status_code == 200
    assert repeated.json()["idempotent"] is True


def test_role_toolset_never_exposes_execute_or_chat_approval():
    from app.services.agent_runtime.state import RuntimeRequestContext
    from app.services.agent_runtime.toolsets import toolset_for_context

    viewer = toolset_for_context(RuntimeRequestContext("u", "viewer", "shared", "c"))
    scientist = toolset_for_context(RuntimeRequestContext("u", "scientist", "shared", "c"))
    assert all(item["effect_kind"] == "read" for item in viewer["tools"])
    assert all(item["effect_kind"] != "execute" for item in scientist["tools"])
    assert scientist["approval_via_chat"] is False


def test_legacy_json_task_import_is_idempotent_and_preserves_source(isolated_sqlite_db, tmp_path, monkeypatch):
    import json

    from app.services.chat import legacy_task_migration

    conversation = assistant_conversations.create_conversation(
        "pytest-approver", workspace_id="shared", title="legacy"
    )
    workflow_dir = tmp_path / "workflow"
    pending_dir = tmp_path / "pending"
    workflow_dir.mkdir()
    pending_dir.mkdir()
    source = workflow_dir / f"{conversation['id']}.json"
    source.write_text(
        json.dumps({
            "active": True,
            "session_id": conversation["id"],
            "source_message": "传代",
            "required_slots": ["strain_id"],
            "collected_slots": {},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(legacy_task_migration, "WORKFLOW_REQUEST_DIR", str(workflow_dir))
    monkeypatch.setattr(legacy_task_migration, "PENDING_FORM_DIR", str(pending_dir))

    first = legacy_task_migration.import_legacy_task_states()
    second = legacy_task_migration.import_legacy_task_states()
    assert first == {"imported": 1, "skipped": 0}
    assert second == {"imported": 0, "skipped": 1}
    assert source.exists()
    tasks = conversation_tasks.list_tasks(
        conversation["id"], owner="pytest-approver", workspace_id="shared"
    )
    assert len(tasks) == 1
    assert tasks[0]["legacy_source"] == str(source.resolve())


def test_missing_workflow_slot_resumes_same_langgraph_task_thread(isolated_sqlite_db, monkeypatch):
    import asyncio
    import sqlite3

    from langgraph.checkpoint.memory import InMemorySaver

    from app.core import database
    from app.schemas.algae import ChatRequest
    from app.services.agent_runtime.checkpoint import set_compiled_graph
    from app.services.agent_runtime.graph_runtime import build_agent_runtime_graph
    from app.services.agent_runtime.state import RuntimeRequestContext
    from app.services.chat import chat_service

    conversation = assistant_conversations.create_conversation(
        "pytest-approver", workspace_id="shared", title="slot resume"
    )
    memory: dict[str, list[dict[str, str]]] = {}
    monkeypatch.setattr(
        chat_service,
        "get_session_memory",
        lambda session_id: list(memory.setdefault(session_id, [{"role": "system", "content": "test"}])),
    )
    monkeypatch.setattr(
        chat_service,
        "save_session_memory",
        lambda session_id, history: memory.__setitem__(session_id, list(history)),
    )
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)
    database.add_algae_strain(
        "Resume_01", "恢复测试藻", "Resume algae", generation_number=2, days_since_last_subculture=8
    )
    context = RuntimeRequestContext(
        owner="pytest-approver",
        role="approver",
        workspace_id="shared",
        conversation_id=conversation["id"],
    )
    set_compiled_graph(build_agent_runtime_graph(checkpointer=InMemorySaver()))
    try:
        first = asyncio.run(chat_service.handle_chat(
            ChatRequest(message="执行传代", session_id=conversation["id"]),
            request_context=context,
        ))
        assert first.agent_output["action"] == "workflow_request_clarification"
        task_id = first.agent_output["task_id"]

        second = asyncio.run(chat_service.handle_chat(
            ChatRequest(message="Resume_01", session_id=conversation["id"]),
            request_context=context,
        ))
        assert second.agent_output["status"] == "pending"
        assert second.agent_output["task_id"] == task_id

        with sqlite3.connect(str(isolated_sqlite_db)) as conn:
            rows = conn.execute(
                "SELECT graph_thread_id, task_id, attempt_no FROM agent_runs WHERE task_id = ? ORDER BY attempt_no",
                (task_id,),
            ).fetchall()
        assert len(rows) == 2
        assert {row[0] for row in rows} == {f"agent-task:{task_id}"}
        assert [row[2] for row in rows] == [1, 2]
    finally:
        set_compiled_graph(None)
