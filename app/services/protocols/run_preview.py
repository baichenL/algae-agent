from __future__ import annotations

from typing import Any

from app.services.labware import get_container_ref
from app.services.protocols.commands import ProtocolCommand, ProtocolRunPreview
from app.services.protocols.models import (
    ExperimentProtocolSpec,
    ProtocolSimulationReport,
    ProtocolValidationReport,
)


OPERATION_LOCATION_HINTS = {
    "check_schedule": "database",
    "sterilize_workspace": "biosafety_cabinet",
    "check_materials": "biosafety_cabinet",
    "dispense_medium": "biosafety_cabinet",
    "transfer_seed": "biosafety_cabinet",
    "close_reactor": "biosafety_cabinet",
    "configure_incubator": "incubator_1",
    "record_experiment": "database",
    "cleanup": "biosafety_cabinet",
    # Reserved for later protocol types.
    "transfer_liquid": "biosafety_cabinet",
    "measure_od": "spectrophotometer_1",
    "centrifuge": "centrifuge_1",
    "weigh": "balance_1",
}


def _container_for(value: Any) -> str | None:
    if not value:
        return None
    container_id = str(value)
    return container_id if get_container_ref(container_id) else container_id


def _command_for_step(index: int, step) -> ProtocolCommand:
    params = dict(step.parameters or {})
    source = _container_for(params.get("source"))
    target = _container_for(
        params.get("target")
        or params.get("reactor")
    )
    return ProtocolCommand(
        command_id=f"cmd-{index:02d}-{step.step_id}",
        step_id=step.step_id,
        operation=step.operation,
        parameters=params,
        device_or_location=OPERATION_LOCATION_HINTS.get(step.operation),
        source_container=source,
        target_container=target,
        requires_human_approval=True,
    )


def build_protocol_run_preview(protocol: ExperimentProtocolSpec) -> ProtocolRunPreview:
    commands = [
        _command_for_step(index, step)
        for index, step in enumerate(protocol.steps, start=1)
    ]
    return ProtocolRunPreview(
        protocol_id=protocol.protocol_id,
        protocol_hash=protocol.protocol_hash,
        commands=commands,
        summary={
            "command_count": len(commands),
            "operations": [command.operation for command in commands],
            "locations": sorted(
                {
                    command.device_or_location
                    for command in commands
                    if command.device_or_location
                }
            ),
        },
    )


def protocol_summary(protocol: ExperimentProtocolSpec) -> dict[str, Any]:
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "template_id": protocol.template_id,
        "template_version": protocol.template_version,
        "target_strain_id": protocol.target_strain_id,
        "expected_generation": protocol.parameters.get("expected_generation"),
        "experiment_type": protocol.experiment_type,
    }


def validation_summary(report: ProtocolValidationReport) -> dict[str, Any]:
    return {
        "valid": report.valid,
        "check_count": len(report.checks),
        "passed_count": len([check for check in report.checks if check.passed]),
        "error_codes": [issue.code for issue in report.errors],
        "warning_codes": [issue.code for issue in report.warnings],
        "validator_version": report.validator_version,
    }


def simulation_summary(report: ProtocolSimulationReport) -> dict[str, Any]:
    return {
        "success": report.success,
        "simulation_id": report.simulation_id,
        "step_count": len(report.step_results),
        "safe_shutdown_triggered": report.safe_shutdown_triggered,
        "failure_code": report.failure_code,
        "simulator_name": report.simulator_name,
        "simulator_version": report.simulator_version,
    }


def preview_rows(
    preview: ProtocolRunPreview,
    simulation_report: ProtocolSimulationReport | None = None,
) -> list[dict[str, Any]]:
    result_by_step = {
        item.step_id: item
        for item in (simulation_report.step_results if simulation_report else [])
    }
    rows = []
    for command in preview.commands:
        result = result_by_step.get(command.step_id)
        rows.append({
            "step_id": command.step_id,
            "operation": command.operation,
            "status": (
                "simulated_ok"
                if result and result.success
                else "simulated_failed"
                if result
                else "planned"
            ),
            "device_or_location": command.device_or_location,
            "source_container": command.source_container,
            "target_container": command.target_container,
            "error_code": result.error_code if result else None,
        })
    return rows
