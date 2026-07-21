import asyncio
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.graph import build_subculture_workflow
from app.hardware import SimulatedHardware
from app.services.workflows.simulation_service import SimulationRunStore


def wait_for_status(store, run_id, statuses, timeout=4):
    deadline = time.time() + timeout
    result = store.get(run_id)
    while result and result["status"] not in set(statuses) and time.time() < deadline:
        time.sleep(0.01)
        result = store.get(run_id)
    return result


def initial_state(hardware, generation=4):
    return {
        "messages": [],
        "execution_mode": "simulation",
        "strain_id": "Chlorella_01",
        "generation_number": generation,
        "inoculation_timestamp": "",
        "source_reactor_id": "Reactor_A",
        "target_reactor_id": "Reactor_B",
        "media_target_volume": 150.0,
        "seed_target_volume": 1.0,
        "days_since_last_subculture": 6,
        "current_step": "START",
        "hardware_logs": [],
        "simulation_events": [],
        "hardware_state": hardware.snapshot(),
        "workflow_status": "IDLE",
        "last_error": None,
    }


def test_stateful_simulator_preserves_liquid_mass_balance():
    hardware = SimulatedHardware(delay_seconds=0)
    before = hardware.snapshot()

    medium = hardware.dispense_medium("Reactor_B", 150.0)
    seed = hardware.transfer_seed("Reactor_A", "Reactor_B", 1.0)

    assert medium.ok is True
    assert seed.ok is True
    after = hardware.snapshot()
    assert after["media_reservoir"]["volume_ml"] == pytest.approx(
        before["media_reservoir"]["volume_ml"] - 150.0
    )
    assert after["source_reactor"]["volume_ml"] == pytest.approx(
        before["source_reactor"]["volume_ml"] - 1.0
    )
    assert after["target_reactor"]["volume_ml"] == pytest.approx(151.0)


def test_langgraph_success_path_uses_hardware_snapshots():
    hardware = SimulatedHardware(delay_seconds=0)
    result = build_subculture_workflow(hardware).invoke(initial_state(hardware))

    assert result["workflow_status"] == "SUCCESS"
    assert result["generation_number"] == 5
    assert result["hardware_state"]["phase"] == "COMPLETED"
    assert result["hardware_state"]["target_reactor"]["volume_ml"] == pytest.approx(151.0)
    assert result["measured_absorbance"] == pytest.approx(0.8)
    assert len(result["simulation_events"]) >= 20


def test_langgraph_hardware_failure_routes_to_safe_shutdown():
    hardware = SimulatedHardware(delay_seconds=0, faults=["pump_a_blocked"])
    result = build_subculture_workflow(hardware).invoke(initial_state(hardware))

    assert result["workflow_status"] == "FAILED"
    assert result["current_step"] == "SafeShutdown"
    assert result["last_error"]["code"] == "MEDIA_PUMP_BLOCKED"
    assert result["hardware_state"]["phase"] == "SAFE_SHUTDOWN"
    assert result["hardware_state"]["media_pump"]["status"] == "OFF"


def test_visual_simulation_run_emits_live_events_and_completes():
    store = SimulationRunStore()
    started = store.create(
        strain_id="Chlorella_01",
        generation_number=7,
        delay_seconds=0,
    )
    run_id = started["run_id"]

    deadline = time.time() + 3
    result = store.get(run_id)
    while result and result["status"] in {"QUEUED", "RUNNING"} and time.time() < deadline:
        time.sleep(0.01)
        result = store.get(run_id)

    assert result is not None
    assert result["status"] == "COMPLETED"
    assert result["workflow_status"] == "SUCCESS"
    assert result["checkpointed"] is True
    assert result["generation_number"] == 8
    assert result["events"]
    assert result["hardware_state"]["target_reactor"]["volume_ml"] == pytest.approx(151.0)


def test_simulation_mode_does_not_update_production_strain(isolated_sqlite_db):
    from app.core.database import get_algae_status
    from app.services.workflows.workflow_service import run_force_subculture_workflow

    before = get_algae_status("Chlorella_01")
    result = asyncio.run(
        run_force_subculture_workflow("Chlorella_01", execution_mode="simulation")
    )
    after = get_algae_status("Chlorella_01")

    assert result["status"] == "success"
    assert result["execution_mode"] == "simulation"
    assert result["persisted"] is False
    assert after["generation_number"] == before["generation_number"]
    assert after["last_subculture_time"] == before["last_subculture_time"]


def test_verification_mode_interrupts_and_resumes_all_manual_tasks_once():
    store = SimulationRunStore(manual_delay_seconds=0)
    started = store.create(
        strain_id="Chlorella_01",
        generation_number=2,
        interaction_mode="verification",
        simulated_absorbance=1.25,
        delay_seconds=0,
    )
    run_id = started["run_id"]
    observed = []

    for expected_task in [
        "load_spectrophotometer",
        "load_liquid_handler",
        "move_to_incubator",
    ]:
        waiting = wait_for_status(store, run_id, {"WAITING_MANUAL"})
        assert waiting["pending_manual_task"]["task_id"] == expected_task
        observed.append(expected_task)
        resolved = store.resolve_manual_task(
            run_id,
            expected_task,
            resolution="completed",
        )
        assert resolved["idempotent"] is False

    completed = wait_for_status(store, run_id, {"COMPLETED", "FAILED"})
    assert completed["status"] == "COMPLETED"
    assert completed["measured_absorbance"] == pytest.approx(1.25)
    assert completed["hardware_state"]["target_reactor"]["volume_ml"] == pytest.approx(151.0)
    manual_actions = [
        event["action"]
        for event in completed["events"]
        if event["device"] == "human_operator" and event["status"] == "completed"
    ]
    assert manual_actions == observed


def test_failed_manual_task_routes_to_safe_shutdown_and_is_idempotent():
    store = SimulationRunStore(manual_delay_seconds=0)
    started = store.create(
        strain_id="Chlorella_01",
        generation_number=2,
        interaction_mode="verification",
        delay_seconds=0,
    )
    run_id = started["run_id"]
    waiting = wait_for_status(store, run_id, {"WAITING_MANUAL"})
    task_id = waiting["pending_manual_task"]["task_id"]

    first = store.resolve_manual_task(
        run_id,
        task_id,
        resolution="failed",
        note="sample was dropped",
    )
    second = store.resolve_manual_task(
        run_id,
        task_id,
        resolution="failed",
        note="duplicate click",
    )
    assert first["idempotent"] is False
    assert second["idempotent"] is True

    failed = wait_for_status(store, run_id, {"FAILED"})
    assert failed["last_error"]["code"] == "MANUAL_TASK_FAILED"
    assert failed["hardware_state"]["phase"] == "SAFE_SHUTDOWN"
    assert not any(event["action"] == "blank" for event in failed["events"])


@pytest.mark.parametrize(
    ("fault", "error_code"),
    [
        ("spectrophotometer_timeout", "SPECTROPHOTOMETER_TIMEOUT"),
        ("blank_calibration_failure", "BLANK_CALIBRATION_FAILED"),
        ("invalid_absorbance", "INVALID_ABSORBANCE"),
        ("liquid_handler_unavailable", "LIQUID_HANDLER_UNAVAILABLE"),
        ("insufficient_medium", "INSUFFICIENT_MEDIUM"),
        ("aspirate_dispense_failure", "ASPIRATE_DISPENSE_FAILURE"),
        ("incubator_setpoint_failure", "INCUBATOR_SETPOINT_FAILURE"),
    ],
)
def test_fault_scenarios_are_reproducible_and_shutdown_safely(fault, error_code):
    store = SimulationRunStore()
    started = store.create(
        strain_id="Chlorella_01",
        generation_number=3,
        interaction_mode="legacy",
        fault_scenario=fault,
        delay_seconds=0,
    )
    failed = wait_for_status(store, started["run_id"], {"FAILED"})
    assert failed["last_error"]["code"] == error_code
    assert failed["hardware_state"]["phase"] == "SAFE_SHUTDOWN"


def test_fifo_queue_waits_for_active_manual_run():
    store = SimulationRunStore(manual_delay_seconds=0)
    first = store.create(
        strain_id="Queue_A",
        generation_number=1,
        interaction_mode="verification",
        delay_seconds=0,
    )
    first_waiting = wait_for_status(store, first["run_id"], {"WAITING_MANUAL"})
    assert first_waiting["queue_position"] == 0

    second = store.create(
        strain_id="Queue_B",
        generation_number=1,
        interaction_mode="legacy",
        delay_seconds=0,
    )
    queued = store.get(second["run_id"])
    assert queued["status"] == "QUEUED"
    assert queued["queue_position"] == 1

    for task_id in [
        "load_spectrophotometer",
        "load_liquid_handler",
        "move_to_incubator",
    ]:
        waiting = wait_for_status(store, first["run_id"], {"WAITING_MANUAL"})
        assert waiting["pending_manual_task"]["task_id"] == task_id
        store.resolve_manual_task(first["run_id"], task_id, resolution="completed")

    assert wait_for_status(store, first["run_id"], {"COMPLETED"})["status"] == "COMPLETED"
    assert wait_for_status(store, second["run_id"], {"COMPLETED"})["status"] == "COMPLETED"


def test_simulation_api_exposes_manual_resolution_and_conflicts(monkeypatch):
    from app.api.workflow import router
    from app.services.workflows import simulation_service

    store = SimulationRunStore(manual_delay_seconds=0)
    monkeypatch.setattr(simulation_service, "simulation_runs", store)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    client = TestClient(app)

    created = client.post(
        "/api/v1/workflow/simulations",
        json={
            "strain_id": "Chlorella_01",
            "generation_number": 1,
            "interaction_mode": "verification",
            "fault_scenario": None,
            "wavelength_nm": 680,
            "simulated_absorbance": 0.8,
        },
    )
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    waiting = wait_for_status(store, run_id, {"WAITING_MANUAL"})
    task_id = waiting["pending_manual_task"]["task_id"]

    resolved = client.post(
        f"/api/v1/workflow/simulations/{run_id}/manual-tasks/{task_id}/resolve",
        json={"resolution": "completed"},
    )
    assert resolved.status_code == 200
    conflicting = client.post(
        f"/api/v1/workflow/simulations/{run_id}/manual-tasks/{task_id}/resolve",
        json={"resolution": "failed"},
    )
    assert conflicting.status_code == 409

    invalid = client.post(
        "/api/v1/workflow/simulations",
        json={
            "strain_id": "Chlorella_01",
            "generation_number": 1,
            "fault_scenario": "unknown_fault",
        },
    )
    assert invalid.status_code == 422
