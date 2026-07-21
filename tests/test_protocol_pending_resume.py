import asyncio
import json
import sqlite3

import pytest

from app.core import database
from app.services.protocols.models import ProtocolSimulationReport
from app.services.strains import strain_service
from app.services.workflows.workflow_approval_service import grant_workflow_approval
from app.services.workflows.workflow_execution_service import execute_workflow_run


def test_validation_failure_does_not_create_pending(isolated_sqlite_db):
    with pytest.raises(ValueError, match="strain_not_found"):
        strain_service.create_pending_workflow_subculture("Missing_01")

    assert database.list_pending_actions(status="pending") == []


def test_simulation_failure_does_not_create_pending(monkeypatch, isolated_sqlite_db):
    def failed_simulation(protocol):
        return ProtocolSimulationReport(
            simulation_id="sim-failed",
            protocol_id=protocol.protocol_id,
            protocol_hash=protocol.protocol_hash,
            success=False,
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            step_results=[],
            final_simulated_state={},
            failure_code="SIMULATION_STEP_FAILED",
            failure_message="failed",
            safe_shutdown_triggered=True,
            simulator_name="test",
            simulator_version="test",
        )

    monkeypatch.setattr(strain_service, "simulate_protocol", failed_simulation)

    with pytest.raises(ValueError, match="SIMULATION_STEP_FAILED"):
        strain_service.create_pending_workflow_subculture("Chlorella_01")

    assert database.list_pending_actions(status="pending") == []


def test_successful_pending_contains_frozen_protocol(isolated_sqlite_db):
    pending_id = strain_service.create_pending_workflow_subculture(
        "Chlorella_01",
        session_id="pending-session",
        agent_run_id="pending-run",
    )
    pending = database.get_pending_action(pending_id)
    frozen = pending["payload"]["protocol"]

    assert pending["status"] == "pending"
    assert pending["payload"]["action_type"] == "execute_experiment_protocol"
    assert frozen["protocol_id"]
    assert frozen["protocol_hash"]
    assert frozen["protocol_spec"]["protocol_hash"] == frozen["protocol_hash"]
    assert frozen["validation_report"]["valid"] is True
    assert frozen["simulation_report"]["success"] is True
    assert frozen["run_preview"]["commands"]
    assert frozen["run_preview"]["commands"][0]["operation"] == "check_schedule"


def test_approval_execution_uses_frozen_protocol_payload(isolated_sqlite_db):
    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")
    approval = grant_workflow_approval(pending_id)
    execution = asyncio.run(execute_workflow_run(approval["run_id"]))
    pending = database.get_pending_action(pending_id)
    frozen = pending["payload"]["protocol"]

    assert approval["status"] == "success"
    assert execution["status"] == "success"
    assert execution["tool_result"]["protocol_hash"] == frozen["protocol_hash"]
    assert execution["persisted"] is False


def test_tampered_protocol_payload_is_rejected_before_approval(isolated_sqlite_db):
    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")
    pending = database.get_pending_action(pending_id)
    payload = pending["payload"]
    payload["protocol"]["protocol_spec"]["parameters"]["media_target_volume"] = 120.0

    with sqlite3.connect(isolated_sqlite_db) as conn:
        conn.execute(
            "UPDATE pending_actions SET payload_json = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), pending_id),
        )
        conn.commit()

    result = grant_workflow_approval(pending_id)

    assert result["status"] == "error"
    assert result["reason"] == "protocol_hash_mismatch"


def test_stale_generation_blocks_old_protocol(isolated_sqlite_db):
    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")
    current = database.get_algae_status("Chlorella_01")
    database.update_algae_status(
        "Chlorella_01",
        int(current["generation_number"]) + 1,
        int(current["days_since_last_subculture"]),
        current["last_subculture_time"],
    )

    result = grant_workflow_approval(pending_id)

    assert result["status"] == "error"
    assert result["reason"] == "stale_generation"


def test_repeated_approval_does_not_create_duplicate_run(isolated_sqlite_db):
    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")
    first = grant_workflow_approval(pending_id)
    second = grant_workflow_approval(pending_id)

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert second["idempotent"] is True
    assert second["run_id"] == first["run_id"]


def test_denied_pending_cannot_execute(isolated_sqlite_db):
    pending_id = strain_service.create_pending_workflow_subculture("Chlorella_01")
    denied = strain_service.deny_pending_action(pending_id)
    result = asyncio.run(strain_service.execute_approved_pending_action(pending_id))

    assert denied["status"] == "success"
    assert result["action"] == "denied"
    assert result["executed"] is False
