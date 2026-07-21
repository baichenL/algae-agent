from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router
from app.core import database
from app.core.db import scientific as scientific_db
from app.services.scientific.demo_data import scientific_demo_file
from app.services.scientific.importer import parse_scientific_dataset
from app.services.scientific.service import run_scientific_task
from app.services.strains import strain_service
from app.services.control_plane import _agent_replans


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _csrf_headers() -> dict[str, str]:
    return {"X-CSRF-Token": "test-csrf"}


def _seed_scientific_run() -> dict:
    content, filename, mapping = scientific_demo_file()
    parsed = parse_scientific_dataset(
        content=content,
        filename=filename,
        strain_id="Chlorella_01",
        mapping=mapping,
    )
    scientific_db.insert_dataset(parsed.dataset, parsed.batches)
    return run_scientific_task(dataset_id=parsed.dataset["id"], offline_replay=True)


def test_dynamic_replan_is_projected_as_business_summary():
    summaries = _agent_replans({
        "trace": {
            "plan": {"mode": "single"},
            "steps": [{
                "step_index": 1,
                "decision": {"action_name": "query_due_subculture"},
                "observation": {"status": "success", "output": {"strains": [{"strain_id": "Chlorella_01"}]}},
                "replan_source": "dynamic_observation",
                "replan_directive": {
                    "is_dynamic": True,
                    "replan_source": "dynamic_observation",
                    "reason": "dynamic_observation_replan",
                    "selected_candidate": {"action_type": "propose_workflow_approval", "target_id": "Chlorella_01", "reason": "one due strain", "confidence": 1.0},
                    "decision": {"action_name": "workflow_subculture"},
                    "policy_boundary": "approval_required",
                },
            }],
        },
        "raw_events": [{"event_type": "agent_replan_directive_created", "payload": {"step_index": 1}, "created_at": "2026-07-17 12:00:00"}],
    })
    assert len(summaries) == 1
    assert summaries[0]["from_version"] == 1
    assert summaries[0]["to_version"] == 2
    assert "Chlorella_01" in " ".join(summaries[0]["key_changes"])
    assert summaries[0]["debug"]["confidence"] == 1.0


def test_v2_session_dashboard_and_run_projection(isolated_sqlite_db):
    run = _seed_scientific_run()
    client = _client()

    session = client.get("/api/v2/session")
    assert session.status_code == 200
    assert session.json()["workspace"]["mode"] == "simulation_only"

    dashboard = client.get("/api/v2/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["counts"]["pending_approvals"] == 1
    with sqlite3.connect(str(isolated_sqlite_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_index WHERE kind = 'scientific'").fetchone()[0] == 1

    canonical_id = f"scientific:{run['id']}"
    detail = client.get(f"/api/v2/runs/{canonical_id}")
    assert detail.status_code == 200
    projected = detail.json()["run"]
    assert projected["status"] == "waiting_approval"
    assert projected["current_action"]["id"] == "approve"
    assert projected["phases"][-3]["key"] == "approval"
    events = client.get(f"/api/v2/runs/{canonical_id}/events").json()["events"]
    assert events
    with sqlite3.connect(str(isolated_sqlite_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_events WHERE run_id = ?", (canonical_id,)).fetchone()[0] == len(events)


def test_scientific_approval_auto_executes(isolated_sqlite_db):
    run = _seed_scientific_run()
    pending_id = run["proposal"]["pending_id"]
    client = _client()

    approved = client.post(
        f"/api/v2/approvals/{pending_id}/decision",
        json={"decision": "approved", "reason": "test approval"},
        headers=_csrf_headers(),
    )
    assert approved.status_code == 200
    after_approval = scientific_db.get_scientific_run(run["id"])
    assert after_approval["status"] == "waiting_approval"
    assert len(after_approval.get("virtual_results") or []) == 1
    assert database.get_pending_action(pending_id)["execution_status"] == "succeeded"


def test_v2_workflow_approval_schedules_execution(isolated_sqlite_db, monkeypatch):
    from app.services import approval_execution_service

    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")
    scheduled = []

    async def fake_execute_workflow_run(run_id: int):
        scheduled.append(run_id)
        return {"status": "success", "run_id": run_id}

    monkeypatch.setattr(approval_execution_service, "execute_workflow_run", fake_execute_workflow_run)
    client = _client()

    approved = client.post(
        f"/api/v2/approvals/{pending_id}/decision",
        json={"decision": "approved", "reason": "auto start"},
        headers=_csrf_headers(),
    )

    assert approved.status_code == 200
    body = approved.json()
    assert body["decision"] == "approved"
    assert body["execution_scheduled"] is True
    assert body["run_id"].startswith("workflow:")
    assert body["operation_id"]
    assert scheduled == [int(body["run_id"].split(":", 1)[1])]


def test_v2_write_requires_csrf_and_tampered_design_is_blocked(isolated_sqlite_db):
    run = _seed_scientific_run()
    pending_id = run["proposal"]["pending_id"]
    canonical_id = f"scientific:{run['id']}"
    client = _client()

    missing_csrf = client.post(
        f"/api/v2/approvals/{pending_id}/decision",
        json={"decision": "approved"},
    )
    assert missing_csrf.status_code == 403

    pending = database.get_pending_action(pending_id)
    payload = pending["payload"]
    payload["data"]["design"]["conditions"][0]["factors"]["tampered"] = 1.0
    with sqlite3.connect(str(isolated_sqlite_db)) as conn:
        conn.execute(
            "UPDATE pending_actions SET payload_json = ? WHERE id = ?",
            (json.dumps(payload), pending_id),
        )

    assert client.post(
        f"/api/v2/approvals/{pending_id}/decision",
        json={"decision": "approved"},
        headers=_csrf_headers(),
    ).status_code == 200
    blocked = client.post(
        f"/api/v2/runs/{canonical_id}/execute",
        json={"idempotency_key": "tampered"},
        headers=_csrf_headers(),
    )
    assert blocked.status_code == 422
    assert blocked.json()["detail"]["reason"] == "design_hash_mismatch"


def test_v1_routes_remain_available_alongside_v2(isolated_sqlite_db):
    client = _client()
    assert client.get("/api/v1/scientific/datasets").status_code == 200
    assert client.get("/api/v2/test-scenarios").status_code == 200


def test_test_lab_scenario_uses_server_generated_isolated_workspace(tmp_path, monkeypatch):
    from app.core.db import connection, control_plane, experiments, pending_actions, rag, reminder_cycles, schema, strains, workflow_runs
    from app.core.workspaces import DATABASE_PATH, WorkspaceContext, workspace_scope
    from app.core import workspaces

    for module in (connection, control_plane, experiments, pending_actions, rag, reminder_cycles, schema, scientific_db, strains, workflow_runs):
        monkeypatch.setattr(module, "DB_PATH", DATABASE_PATH)
    monkeypatch.setattr(workspaces, "TEST_WORKSPACE_ROOT", tmp_path / "test_workspaces")
    shared = WorkspaceContext(id="shared-test", name="shared", db_path=str(tmp_path / "shared.sqlite3"))
    client = _client()

    with workspace_scope(shared):
        schema.init_db()
        response = client.post(
            "/api/v2/test-scenarios/scientific-two-cycle/runs",
            json={"parameters": {"use_demo_dataset": True}},
            headers=_csrf_headers(),
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["workspace"]["id"].startswith("test-scientific-two-cycle-")
        assert payload["workspace"]["seed"] == 2025
        assert scientific_db.list_scientific_runs() == []

        detail = client.get(f"/api/v2/runs/{payload['run_id']}")
        assert detail.status_code == 200
        assert detail.json()["run"]["workspace_id"] == payload["workspace"]["id"]

        deleted = client.delete(
            f"/api/v2/test-workspaces/{payload['workspace']['id']}",
            headers=_csrf_headers(),
        )
        assert deleted.status_code == 200


def test_manual_subculture_confirmation_updates_laboratory_fact_once(isolated_sqlite_db):
    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")
    before = database.get_algae_status("Chlorella_01")
    client = _client()
    approved = client.post(
        f"/api/v2/approvals/{pending_id}/decision",
        json={"decision": "approved"},
        headers=_csrf_headers(),
    )
    run_id = approved.json()["run_id"]
    simulation_events = client.get(f"/api/v2/runs/{run_id}/events").json()["events"]
    event_types = {item["event_type"] for item in simulation_events}
    assert "simulation_started" in event_types
    assert "simulation_snapshot" in event_types
    assert "simulation_completed" in event_types
    completed_steps = {
        item["payload"].get("step")
        for item in simulation_events
        if item["event_type"] == "simulation_step_completed"
    }
    assert completed_steps == {
        "CheckSchedule",
        "ManualLoadSpectrophotometer",
        "BlankSpectrophotometer",
        "MeasureAbsorbance",
        "ManualLoadLiquidHandler",
        "LoadMaterials",
        "DispenseMedium",
        "TransferSeedCulture",
        "MixAndSeal",
        "ManualMoveToIncubator",
        "MoveToIncubator",
        "RecordExperiment",
        "CleanupWorkspace",
    }
    projected = client.get(f"/api/v2/runs/{run_id}").json()["run"]
    assert projected["simulation"]["progress"] == 1.0
    assert projected["simulation"]["replay_available"] is True
    after_simulation = database.get_algae_status("Chlorella_01")
    assert after_simulation["generation_number"] == before["generation_number"]

    payload = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "note": "manual bench transfer",
        "idempotency_key": "manual-commit-001",
    }
    committed = client.post(
        f"/api/v2/runs/{run_id}/manual-commit", json=payload, headers=_csrf_headers(),
    )
    assert committed.status_code == 200
    after_commit = database.get_algae_status("Chlorella_01")
    assert after_commit["generation_number"] == before["generation_number"] + 1
    assert after_commit["days_since_last_subculture"] == 0
    assert committed.json()["run"]["commit_source"] == "manual_confirmation"
    assert committed.json()["run"]["physical_execution"] is False

    repeated = client.post(
        f"/api/v2/runs/{run_id}/manual-commit", json=payload, headers=_csrf_headers(),
    )
    assert repeated.status_code == 200
    assert repeated.json()["idempotent"] is True
    assert database.get_algae_status("Chlorella_01")["generation_number"] == after_commit["generation_number"]


def test_resource_strain_request_executes_after_approval(isolated_sqlite_db):
    client = _client()
    created = client.post(
        "/api/v2/resources/strains/requests",
        json={"operation": "add", "data": {"strain_id": "UI_01", "name_cn": "界面品系", "name_en": "UI strain", "generation_number": 1, "days_since_last_subculture": 0}},
        headers=_csrf_headers(),
    )
    assert created.status_code == 201
    pending_id = created.json()["pending_id"]
    approved = client.post(
        f"/api/v2/approvals/{pending_id}/decision",
        json={"decision": "approved"},
        headers=_csrf_headers(),
    )
    assert approved.status_code == 200
    assert database.get_algae_status("UI_01")["name_en"] == "UI strain"
    assert database.get_pending_action(pending_id)["execution_status"] == "succeeded"

    updated = client.post(
        "/api/v2/resources/strains/requests",
        json={"operation": "update", "data": {"strain_id": "UI_01", "generation_number": 4, "days_since_last_subculture": 7}},
        headers=_csrf_headers(),
    )
    assert updated.status_code == 201
    update_pending_id = updated.json()["pending_id"]
    update_approved = client.post(
        f"/api/v2/approvals/{update_pending_id}/decision",
        json={"decision": "approved"},
        headers=_csrf_headers(),
    )
    assert update_approved.status_code == 200
    status = database.get_algae_status("UI_01")
    assert status["generation_number"] == 4
    assert status["days_since_last_subculture"] == 7


def test_assistant_conversations_are_persisted_and_archivable(isolated_sqlite_db):
    client = _client()
    created = client.post(
        "/api/v2/assistant/conversations", json={}, headers=_csrf_headers(),
    )
    assert created.status_code == 201
    conversation_id = created.json()["conversation"]["id"]
    response = client.post(
        f"/api/v2/assistant/conversations/{conversation_id}/messages",
        json={"content": "你能完成哪些工作？", "client_message_id": "message-0001"},
        headers=_csrf_headers(),
    )
    assert response.status_code == 202
    assert response.json()["operation_id"]
    operation = client.get(f"/api/v2/operations/{response.json()['operation_id']}")
    assert operation.status_code == 200
    assert operation.json()["operation"]["status"] == "succeeded"
    history = client.get(f"/api/v2/assistant/conversations/{conversation_id}/messages").json()
    assert [item["role"] for item in history["messages"]] == ["user", "assistant"]
    assert history["messages"][1]["status"] == "completed"
    assert history["conversation"]["title"].startswith("你能完成哪些工作")
    archived = client.patch(
        f"/api/v2/assistant/conversations/{conversation_id}",
        json={"status": "archived"},
        headers=_csrf_headers(),
    )
    assert archived.status_code == 200
    active_ids = {item["id"] for item in client.get("/api/v2/assistant/conversations").json()["conversations"]}
    assert conversation_id not in active_ids


def test_scientific_test_scenario_requires_explicit_dataset_source(isolated_sqlite_db):
    client = _client()
    missing = client.post(
        "/api/v2/test-scenarios/scientific-two-cycle/runs",
        json={"parameters": {}},
        headers=_csrf_headers(),
    )
    assert missing.status_code == 422


def test_knowledge_upload_archive_and_restore_controls_retrieval(isolated_sqlite_db, tmp_path, monkeypatch):
    from app.core.db import rag as rag_db
    from app.services.rag import knowledge_asset_service

    monkeypatch.setattr(knowledge_asset_service, "KNOWLEDGE_UPLOAD_ROOT", tmp_path / "knowledge")
    client = _client()
    uploaded = client.post(
        "/api/v2/knowledge/sources",
        files={"file": ("manual.txt", b"photobioreactor uniquealpha calibration procedure", "text/plain")},
        data={"doc_type": "manual", "title": "Calibration SOP", "language": "en"},
        headers=_csrf_headers(),
    )
    assert uploaded.status_code == 202
    source_id = uploaded.json()["source"]["source_id"]
    source = database.get_rag_knowledge_source(source_id)
    assert source["ingestion_status"] == "indexed"
    assert source["doc_type"] == "manual"
    assert rag_db.search_rag_chunks("uniquealpha", top_k=5)

    archived = client.post(
        f"/api/v2/knowledge/sources/{source_id}/archive", headers=_csrf_headers(),
    )
    assert archived.status_code == 200
    assert rag_db.search_rag_chunks("uniquealpha", top_k=5) == []

    restored = client.post(
        f"/api/v2/knowledge/sources/{source_id}/restore", headers=_csrf_headers(),
    )
    assert restored.status_code == 200
    assert rag_db.search_rag_chunks("uniquealpha", top_k=5)


def test_test_workspace_email_is_written_to_outbox_without_smtp(tmp_path, monkeypatch):
    import smtplib

    from app.core.workspaces import WorkspaceContext, workspace_scope
    from app.models.email_schema import EmailSendRequest
    from app.services.email.email_service import send_email

    workspace = WorkspaceContext(
        id="test-email",
        name="email sandbox",
        db_path=str(tmp_path / "workspace.sqlite3"),
        email_mode="test_outbox",
    )
    monkeypatch.setattr(smtplib, "SMTP", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SMTP called")))
    with workspace_scope(workspace):
        result = send_email(EmailSendRequest(subject="test", body="body", recipients=["lab@example.test"]))
    assert result.status == "success"
    assert result.sent is False
    assert (tmp_path / "test_outbox.jsonl").read_text(encoding="utf-8").count("lab@example.test") == 1


def test_v2_browser_session_role_and_csrf_matrix(monkeypatch):
    monkeypatch.setenv("ALGAE_AUTH_MODE", "required")
    monkeypatch.setenv(
        "ALGAE_API_KEYS_JSON",
        json.dumps({
            "viewer-key": {"name": "viewer", "role": "viewer"},
            "scientist-key": {"name": "scientist", "role": "scientist"},
        }),
    )
    viewer = _client()
    login = viewer.post("/api/v2/session/login", json={}, headers={"X-API-Key": "viewer-key"})
    assert login.status_code == 200
    csrf = login.json()["csrf_token"]
    assert viewer.post(
        "/api/v2/runs", json={"kind": "simulation", "input": {}}, headers={"X-CSRF-Token": csrf},
    ).status_code == 403

    scientist = _client()
    login = scientist.post("/api/v2/session/login", json={}, headers={"X-API-Key": "scientist-key"})
    csrf = login.json()["csrf_token"]
    assert scientist.post(
        "/api/v2/runs", json={"kind": "simulation", "input": {}}, headers={"X-CSRF-Token": csrf},
    ).status_code == 202
    assert scientist.post(
        "/api/v2/approvals/1/decision", json={"decision": "approved"}, headers={"X-CSRF-Token": csrf},
    ).status_code == 403
