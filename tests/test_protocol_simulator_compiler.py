import pytest

from app.core import database
from app.hardware import SimulatedHardware
from app.services.protocols import (
    build_subculture_protocol,
    compile_protocol_to_workflow,
    compute_protocol_hash,
    simulate_protocol,
)
from app.services.protocols.models import ExperimentProtocolSpec


def _protocol():
    return build_subculture_protocol(
        strain_id="Chlorella_01",
        strain_snapshot=database.get_algae_status("Chlorella_01"),
        session_id="sim-session",
        agent_run_id="sim-run",
        source_message="run subculture",
    )


def test_protocol_simulation_success_does_not_mutate_database(isolated_sqlite_db):
    protocol = _protocol()
    before = database.get_algae_status("Chlorella_01")

    report = simulate_protocol(protocol)
    after = database.get_algae_status("Chlorella_01")

    assert report.success is True
    assert report.step_results
    assert after["generation_number"] == before["generation_number"]
    assert after["last_subculture_time"] == before["last_subculture_time"]


def test_protocol_simulation_failure_stops_and_safe_shutdowns(isolated_sqlite_db):
    report = simulate_protocol(
        _protocol(),
        hardware=SimulatedHardware(delay_seconds=0, faults=["pump_a_blocked"]),
    )

    operations = [step.operation for step in report.step_results]
    assert report.success is False
    assert report.failure_code == "MEDIA_PUMP_BLOCKED"
    assert report.safe_shutdown_triggered is True
    assert "transfer_seed" not in operations
    assert report.final_simulated_state["phase"] == "SAFE_SHUTDOWN"


def test_compiler_binds_protocol_to_existing_workflow_state(isolated_sqlite_db):
    protocol = _protocol()

    compiled = compile_protocol_to_workflow(
        protocol,
        strain_snapshot=database.get_algae_status("Chlorella_01"),
    )

    assert compiled["protocol_id"] == protocol.protocol_id
    assert compiled["protocol_hash"] == protocol.protocol_hash
    assert compiled["initial_state"]["media_target_volume"] == pytest.approx(150.0)
    assert compiled["initial_state"]["protocol_hash"] == protocol.protocol_hash


def test_compiler_rejects_unknown_operation(isolated_sqlite_db):
    protocol = ExperimentProtocolSpec.from_dict(_protocol().to_dict())
    protocol.steps[0] = type(protocol.steps[0])(
        **{**protocol.steps[0].to_dict(), "operation": "run_python"}
    )
    protocol.protocol_hash = compute_protocol_hash(protocol)

    with pytest.raises(ValueError, match="PROTOCOL_OPERATION_NOT_ALLOWED"):
        compile_protocol_to_workflow(
            protocol,
            strain_snapshot=database.get_algae_status("Chlorella_01"),
        )
