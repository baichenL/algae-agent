from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.memory import router as memory_router
from app.core.db.connection import connect
from app.services.agent_runtime.state import ContextBundle, ContextSlice
from app.services.user_memory.retriever import retrieve_user_memories
from app.services.user_memory.migration import import_legacy_user_memories
from app.services.user_memory.service import process_completed_turn
from app.services.user_memory.store import (
    apply_memory,
    create_candidate,
    delete_memory,
    get_memory,
    get_settings,
    list_candidates,
    list_memories,
    update_memory,
    update_settings,
)


def test_settings_are_opt_in_and_scoped(isolated_sqlite_db):
    assert get_settings("alice", "lab-a")["enabled"] is False
    updated = update_settings("alice", "lab-a", enabled=True, auto_write_low_risk=False)
    assert updated["enabled"] is True
    assert updated["auto_write_low_risk"] is False
    assert get_settings("alice", "lab-b")["enabled"] is False
    assert get_settings("bob", "lab-a")["enabled"] is False


def test_singleton_update_supersedes_and_retrieval_is_isolated(isolated_sqlite_db):
    action, first = apply_memory(
        owner_id="alice", workspace_id="lab", memory_type="preference",
        predicate="response.language", value="English", actor="alice", confirmation=True,
    )
    assert action == "ADD"
    action, second = apply_memory(
        owner_id="alice", workspace_id="lab", memory_type="preference",
        predicate="response.language", value="中文", actor="alice", confirmation=True,
    )
    assert action == "UPDATE"
    assert second["revision"] == 2
    assert get_memory(first["id"], owner_id="alice", workspace_id="lab")["status"] == "superseded"
    assert list_memories(owner_id="alice", workspace_id="lab", status="active")[0]["value"] == "中文"

    selected = retrieve_user_memories(
        owner_id="alice", workspace_id="lab", query="请用中文回答",
        route_kind="chat", record_usage=False,
    )
    assert [item["id"] for item in selected] == [second["id"]]
    assert retrieve_user_memories(
        owner_id="bob", workspace_id="lab", query="中文",
        route_kind="chat", record_usage=False,
    ) == []


def test_sensitive_values_never_persist(isolated_sqlite_db):
    try:
        apply_memory(
            owner_id="alice", workspace_id="lab", memory_type="note",
            predicate="general.note", value="api_key=sk-secret-value", actor="alice",
        )
    except ValueError as exc:
        assert str(exc) == "prohibited_sensitive_memory"
    else:
        raise AssertionError("sensitive memory should be rejected")
    with connect(row_factory=True) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM user_memories").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM user_memory_events").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM user_memories_fts").fetchone()["n"] == 0
    try:
        create_candidate(
            owner_id="alice", workspace_id="lab", memory_type="note", predicate="general.note",
            value="password: do-not-save", confidence=0.9, sensitivity="low", reason="bad",
            evidence="bad", model_name="test", source_conversation_id="c1",
            source_message_id=99, source_run_id=None,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("sensitive candidate should be rejected")
    assert list_candidates(owner_id="alice", workspace_id="lab") == []


def test_candidate_idempotency_and_completed_turn_failure_is_non_fatal(isolated_sqlite_db, monkeypatch):
    kwargs = dict(
        owner_id="alice", workspace_id="lab", memory_type="preference",
        predicate="report.format", value="Markdown", confidence=0.7, sensitivity="low",
        reason="explicit", evidence="Markdown", model_name="fake",
        source_conversation_id="c1", source_message_id=10, source_run_id="agent:r1",
    )
    one = create_candidate(**kwargs)
    two = create_candidate(**kwargs)
    assert one["id"] == two["id"]
    assert len(list_candidates(owner_id="alice", workspace_id="lab")) == 1

    update_settings("alice", "lab", enabled=True, auto_write_low_risk=True)
    monkeypatch.setenv("USER_MEMORY_ENABLED", "true")
    monkeypatch.setattr("app.services.user_memory.service.extract_candidates", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    result = process_completed_turn(
        owner_id="alice", workspace_id="lab", conversation_id="c1", user_message_id=11,
        user_text="Use Markdown", assistant_text="Okay",
    )
    assert result["status"] == "failed"


def test_trace_redacts_memory_value(isolated_sqlite_db):
    secret_preference = "a-user-value-that-must-not-enter-trace"
    empty = lambda name, authority: ContextSlice(name=name, source="test", authority=authority, summary={}, payload={})
    bundle = ContextBundle(
        fact_context=empty("fact_context", "database_current_state"),
        session_context=empty("session_context", "conversation_history"),
        rag_context=empty("rag_context", "knowledge_only"),
        tool_context=empty("tool_context", "current_run_observation"),
        policy_context=empty("policy_context", "execution_boundary"),
        user_memory_context=ContextSlice(
            name="user_memory_context", source="user_memories", authority="user_preference_advisory",
            summary={"memory_count": 1}, payload={"memories": [{"id": 7, "predicate": "general.note", "revision": 1, "score": 0.9, "value": secret_preference}]},
        ),
    )
    assert secret_preference not in str(bundle.to_event_payload())
    assert secret_preference in str(bundle.to_model_payload())


def test_delete_removes_body_fts_and_embedding_but_keeps_tombstone(isolated_sqlite_db):
    _, memory = apply_memory(
        owner_id="alice", workspace_id="lab", memory_type="preference",
        predicate="units.preferred", value="SI", actor="alice",
    )
    with connect() as conn:
        conn.execute(
            "INSERT INTO user_memory_embeddings VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory["id"], "hash", "test", 1, b"0", "now", "now"),
        )
        conn.commit()
    delete_memory(memory["id"], owner_id="alice", workspace_id="lab", actor="alice")
    assert get_memory(memory["id"], owner_id="alice", workspace_id="lab") is None
    with connect(row_factory=True) as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM user_memories_fts WHERE memory_id = ?", (memory["id"],)).fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM user_memory_embeddings WHERE memory_id = ?", (memory["id"],)).fetchone()["n"] == 0
        event = conn.execute("SELECT * FROM user_memory_events WHERE event_type = 'deleted'").fetchone()
        assert conn.execute("SELECT COUNT(*) AS n FROM user_memory_candidates WHERE applied_memory_id = ?", (memory["id"],)).fetchone()["n"] == 0
    assert event is not None
    assert "SI" not in str(dict(event))


def test_memory_api_crud_and_revision_conflict(isolated_sqlite_db):
    app = FastAPI()
    app.include_router(memory_router)
    client = TestClient(app)
    headers = {"X-CSRF-Token": "test-csrf"}

    assert client.get("/api/v2/memory-settings").json()["settings"]["enabled"] is False
    response = client.patch(
        "/api/v2/memory-settings", headers=headers,
        json={"enabled": True, "auto_write_low_risk": True},
    )
    assert response.status_code == 200
    created = client.post(
        "/api/v2/memories", headers=headers,
        json={"memory_type": "preference", "predicate": "response.detail_level", "value": "concise"},
    )
    assert created.status_code == 200
    memory = created.json()["memory"]
    changed = client.patch(
        f"/api/v2/memories/{memory['id']}", headers=headers,
        json={"expected_revision": memory["revision"], "value": "detailed"},
    )
    assert changed.status_code == 200
    conflict = client.patch(
        f"/api/v2/memories/{changed.json()['memory']['id']}", headers=headers,
        json={"expected_revision": memory["revision"], "value": "concise"},
    )
    assert conflict.status_code == 409
    assert client.get("/api/v2/memories?status=active").json()["memories"][0]["value"] == "detailed"


def test_legacy_migration_is_idempotent_and_filters_sensitive_content(isolated_sqlite_db):
    with connect() as conn:
        conn.execute(
            "INSERT INTO assistant_conversations (id, owner, workspace_id, title, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("legacy-c1", "alice", "lab", "legacy", "active", "now", "now"),
        )
        conn.execute(
            "INSERT INTO agent_memories (scope, content, confidence, status) VALUES (?, ?, ?, ?)",
            ("user:legacy-c1", "Prefer concise Markdown reports", 0.7, "active"),
        )
        conn.execute(
            "INSERT INTO agent_memories (scope, content, confidence, status) VALUES (?, ?, ?, ?)",
            ("user:legacy-c1", "password: never-persist-this", 0.7, "active"),
        )
        conn.commit()
    first = import_legacy_user_memories()
    second = import_legacy_user_memories()
    candidates = list_candidates(owner_id="alice", workspace_id="lab")
    assert first == {"imported": 1, "unresolved": 1}
    assert second == {"imported": 1, "unresolved": 1}
    assert len(candidates) == 1
    assert "never-persist-this" not in str(candidates)


def test_memory_eval_fixture_has_three_tiers_and_sixty_fixed_cases():
    path = Path(__file__).parent / "fixtures" / "user_memory_eval_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    tiers = {name: [item for item in cases if item["tier"] == name] for name in (
        "basic_recall", "multi_conversation", "proactive_service",
    )}
    assert len(cases) == 60
    assert all(len(items) == 20 for items in tiers.values())
    assert len({item["id"] for item in cases}) == 60
