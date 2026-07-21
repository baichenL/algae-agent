from __future__ import annotations

from typing import Any

from app.services.protocols.hashing import verify_protocol_hash
from app.services.protocols.models import ExperimentProtocolSpec
from app.services.protocols.templates import SUPPORTED_PROTOCOL_OPERATIONS, get_subculture_template


def compile_protocol_to_workflow(
    protocol: ExperimentProtocolSpec,
    *,
    strain_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    if protocol.experiment_type != "subculture":
        raise ValueError("PROTOCOL_OPERATION_NOT_ALLOWED")
    if not verify_protocol_hash(protocol):
        raise ValueError("PROTOCOL_HASH_MISMATCH")
    if not strain_snapshot:
        raise ValueError("TARGET_STRAIN_NOT_FOUND")
    expected_generation = int(protocol.parameters.get("expected_generation") or 0)
    current_generation = int(strain_snapshot.get("generation_number") or 0)
    if expected_generation != current_generation:
        raise ValueError("TARGET_STATE_STALE")

    template = get_subculture_template()
    expected_steps = [step.step_id for step in sorted(template.steps, key=lambda item: item.order)]
    actual_steps = [step.step_id for step in protocol.steps]
    if expected_steps != actual_steps:
        raise ValueError("PROTOCOL_STEP_ORDER_INVALID")
    for step in protocol.steps:
        if step.operation not in SUPPORTED_PROTOCOL_OPERATIONS:
            raise ValueError("PROTOCOL_OPERATION_NOT_ALLOWED")

    params = protocol.parameters
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "workflow_name": "subculture",
        "operation_sequence": [
            {"step_id": step.step_id, "operation": step.operation}
            for step in protocol.steps
        ],
        "initial_state": {
            "messages": [],
            "execution_mode": "simulation",
            "strain_id": protocol.target_strain_id,
            "generation_number": expected_generation,
            "inoculation_timestamp": "",
            "source_reactor_id": params["source_reactor_id"],
            "target_reactor_id": params["target_reactor_id"],
            "media_target_volume": float(params["media_target_volume"]),
            "seed_target_volume": float(params["seed_target_volume"]),
            "days_since_last_subculture": int(params["days_since_last_subculture"]),
            "current_step": "START",
            "hardware_logs": [],
            "simulation_events": [],
            "hardware_state": {},
            "last_error": None,
            "workflow_status": "IDLE",
            "protocol_id": protocol.protocol_id,
            "protocol_hash": protocol.protocol_hash,
        },
    }
