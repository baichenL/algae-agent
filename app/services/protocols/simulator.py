from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from app.hardware import HardwareResult, SimulatedHardware
from app.services.protocols.models import (
    ExperimentProtocolSpec,
    ProtocolSimulationReport,
    ProtocolSimulationStepResult,
)


SIMULATOR_NAME = "SimulatedHardware"
SIMULATOR_VERSION = "protocol_simulator_v1"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _success_result(
    *,
    step_id: str,
    operation: str,
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    observation: dict[str, Any],
) -> ProtocolSimulationStepResult:
    return ProtocolSimulationStepResult(
        step_id=step_id,
        operation=operation,
        success=True,
        observation=observation,
        error_code=None,
        error_message=None,
        state_before=state_before,
        state_after=state_after,
    )


def _hardware_step(
    hardware: SimulatedHardware,
    operation: str,
    parameters: dict[str, Any],
) -> HardwareResult:
    if operation == "sterilize_workspace":
        return hardware.sterilize_workspace(float(parameters["duration_min"]))
    if operation == "check_materials":
        return hardware.check_materials(float(parameters["expected_media_ml"]))
    if operation == "dispense_medium":
        return hardware.dispense_medium(
            target=str(parameters["target"]),
            volume_ml=float(parameters["volume_ml"]),
        )
    if operation == "transfer_seed":
        return hardware.transfer_seed(
            source=str(parameters["source"]),
            target=str(parameters["target"]),
            volume_ml=float(parameters["volume_ml"]),
        )
    if operation == "close_reactor":
        return hardware.close_reactor(str(parameters["reactor"]))
    if operation == "configure_incubator":
        return hardware.configure_incubator(
            temperature_c=float(parameters["temperature_c"]),
            light_lux=float(parameters["light_lux"]),
        )
    if operation == "cleanup":
        return hardware.cleanup()
    raise ValueError(f"unsupported_simulation_operation:{operation}")


def simulate_protocol(
    protocol: ExperimentProtocolSpec,
    *,
    hardware: SimulatedHardware | None = None,
) -> ProtocolSimulationReport:
    simulator = hardware or SimulatedHardware(delay_seconds=0)
    simulation_id = str(uuid.uuid4())
    started_at = _now()
    step_results: list[ProtocolSimulationStepResult] = []
    generation = int(protocol.parameters.get("expected_generation") or 0)
    safe_shutdown_triggered = False
    failure_code = None
    failure_message = None

    for step in protocol.steps:
        state_before = simulator.snapshot()
        try:
            if step.operation == "check_schedule":
                days = int(step.parameters.get("days_since_last_subculture") or 0)
                minimum = int(step.parameters.get("minimum_days_since_last_subculture") or 0)
                step_results.append(
                    _success_result(
                        step_id=step.step_id,
                        operation=step.operation,
                        state_before=state_before,
                        state_after=simulator.snapshot(),
                        observation={
                            "days_since_last_subculture": days,
                            "minimum_days_since_last_subculture": minimum,
                            "allowed": days >= minimum,
                        },
                    )
                )
                continue
            if step.operation == "record_experiment":
                generation = int(step.parameters.get("expected_next_generation") or generation + 1)
                step_results.append(
                    _success_result(
                        step_id=step.step_id,
                        operation=step.operation,
                        state_before=state_before,
                        state_after=simulator.snapshot(),
                        observation={"next_generation": generation},
                    )
                )
                continue

            result = _hardware_step(simulator, step.operation, step.parameters)
            state_after = result.state
            step_results.append(
                ProtocolSimulationStepResult(
                    step_id=step.step_id,
                    operation=step.operation,
                    success=result.ok,
                    observation={
                        "device": result.device,
                        "action": result.action,
                        "message": result.message,
                        "event_count": len(result.events),
                    },
                    error_code=result.error_code,
                    error_message=None if result.ok else result.message,
                    state_before=state_before,
                    state_after=state_after,
                )
            )
            if not result.ok:
                failure_code = result.error_code or "SIMULATION_STEP_FAILED"
                failure_message = result.message
                simulator.safe_shutdown(result.message)
                safe_shutdown_triggered = True
                break
        except Exception as exc:
            failure_code = "SIMULATION_STEP_FAILED"
            failure_message = str(exc)
            shutdown = simulator.safe_shutdown(str(exc))
            safe_shutdown_triggered = True
            step_results.append(
                ProtocolSimulationStepResult(
                    step_id=step.step_id,
                    operation=step.operation,
                    success=False,
                    observation={},
                    error_code=failure_code,
                    error_message=failure_message,
                    state_before=state_before,
                    state_after=shutdown.state,
                )
            )
            break

    success = failure_code is None
    return ProtocolSimulationReport(
        simulation_id=simulation_id,
        protocol_id=protocol.protocol_id,
        protocol_hash=protocol.protocol_hash,
        success=success,
        started_at=started_at,
        completed_at=_now(),
        step_results=step_results,
        final_simulated_state=simulator.snapshot(),
        failure_code=failure_code,
        failure_message=failure_message,
        safe_shutdown_triggered=safe_shutdown_triggered,
        simulator_name=SIMULATOR_NAME,
        simulator_version=SIMULATOR_VERSION,
    )
