from __future__ import annotations

from typing import Any

from app.services.protocols.hashing import verify_protocol_hash
from app.services.protocols.models import (
    ExperimentProtocolSpec,
    ProtocolSimulationReport,
    ProtocolValidationReport,
)
from app.services.protocols.commands import ProtocolRunPreview
from app.services.protocols.run_preview import (
    preview_rows,
    protocol_summary,
    simulation_summary,
    validation_summary,
)


def build_protocol_payload(
    *,
    protocol: ExperimentProtocolSpec,
    validation_report: ProtocolValidationReport,
    simulation_report: ProtocolSimulationReport,
    run_preview: ProtocolRunPreview | None = None,
    expected_generation: int,
    target_strain_id: str,
) -> dict[str, Any]:
    payload = {
        "action_type": "execute_experiment_protocol",
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "protocol_hash": protocol.protocol_hash,
        "protocol_spec": protocol.to_dict(),
        "validation_report": validation_report.to_dict(),
        "simulation_report": simulation_report.to_dict(),
        "expected_generation": int(expected_generation),
        "target_strain_id": target_strain_id,
    }
    if run_preview is not None:
        payload["run_preview"] = run_preview.to_dict()
    return payload


def get_frozen_protocol_payload(pending_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pending_payload:
        return None
    frozen = pending_payload.get("protocol")
    return frozen if isinstance(frozen, dict) else None


def load_frozen_protocol(pending_payload: dict[str, Any] | None) -> ExperimentProtocolSpec | None:
    frozen = get_frozen_protocol_payload(pending_payload)
    if not frozen:
        return None
    spec = frozen.get("protocol_spec")
    if not isinstance(spec, dict):
        return None
    return ExperimentProtocolSpec.from_dict(spec)


def verify_frozen_protocol_payload(pending_payload: dict[str, Any] | None) -> dict[str, Any]:
    frozen = get_frozen_protocol_payload(pending_payload)
    if not frozen:
        return {"valid": True, "reason": "legacy_workflow_pending"}
    protocol = load_frozen_protocol(pending_payload)
    if protocol is None:
        return {"valid": False, "reason": "protocol_missing"}
    expected_hash = frozen.get("protocol_hash")
    if protocol.protocol_hash != expected_hash:
        return {
            "valid": False,
            "reason": "protocol_hash_mismatch",
            "protocol_id": frozen.get("protocol_id"),
            "protocol_hash": expected_hash,
        }
    if not verify_protocol_hash(protocol, expected_hash):
        return {
            "valid": False,
            "reason": "protocol_hash_mismatch",
            "protocol_id": protocol.protocol_id,
            "protocol_hash": expected_hash,
        }
    return {
        "valid": True,
        "reason": "ok",
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.protocol_hash,
        "expected_generation": frozen.get("expected_generation"),
        "target_strain_id": frozen.get("target_strain_id"),
    }


def build_protocol_response_fields(pending_payload: dict[str, Any] | None) -> dict[str, Any]:
    frozen = get_frozen_protocol_payload(pending_payload)
    if not frozen:
        return {
            "protocol_summary": {},
            "validation_summary": {},
            "simulation_summary": {},
            "run_preview": [],
        }
    protocol = load_frozen_protocol(pending_payload)
    validation_report = (
        ProtocolValidationReport.from_dict(frozen.get("validation_report") or {})
        if frozen.get("validation_report")
        else None
    )
    simulation_report = (
        ProtocolSimulationReport.from_dict(frozen.get("simulation_report") or {})
        if frozen.get("simulation_report")
        else None
    )
    run_preview = (
        ProtocolRunPreview.from_dict(frozen.get("run_preview") or {})
        if frozen.get("run_preview")
        else None
    )
    return {
        "protocol_summary": protocol_summary(protocol) if protocol else {},
        "validation_summary": validation_summary(validation_report) if validation_report else {},
        "simulation_summary": simulation_summary(simulation_report) if simulation_report else {},
        "run_preview": (
            preview_rows(run_preview, simulation_report)
            if run_preview
            else []
        ),
    }
