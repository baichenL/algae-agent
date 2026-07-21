from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.services.protocols.hashing import compute_protocol_hash
from app.services.protocols.models import ExperimentProtocolSpec, ProtocolStep, ProtocolTemplate
from app.services.protocols.templates import (
    DEFAULT_SUBCULTURE_PARAMETERS,
    get_subculture_template,
)


class ProtocolBuildError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        missing_fields: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
        self.missing_fields = missing_fields or []
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "missing_fields": self.missing_fields,
            "details": self.details,
        }


def _required(value: Any, field_name: str, missing: list[str]) -> Any:
    if value is None or value == "":
        missing.append(field_name)
    return value


def _step_parameters(operation: str, parameters: dict[str, Any]) -> dict[str, Any]:
    if operation == "check_schedule":
        return {
            "days_since_last_subculture": parameters["days_since_last_subculture"],
            "minimum_days_since_last_subculture": 6,
        }
    if operation == "sterilize_workspace":
        return {"duration_min": 30.0}
    if operation == "check_materials":
        return {"expected_media_ml": parameters["media_target_volume"]}
    if operation == "dispense_medium":
        return {
            "target": parameters["target_reactor_id"],
            "volume_ml": parameters["media_target_volume"],
        }
    if operation == "transfer_seed":
        return {
            "source": parameters["source_reactor_id"],
            "target": parameters["target_reactor_id"],
            "volume_ml": parameters["seed_target_volume"],
        }
    if operation == "close_reactor":
        return {"reactor": parameters["target_reactor_id"]}
    if operation == "configure_incubator":
        return {
            "temperature_c": parameters["temperature_c"],
            "light_lux": parameters["light_lux"],
        }
    if operation == "record_experiment":
        return {"expected_next_generation": int(parameters["expected_generation"]) + 1}
    if operation == "cleanup":
        return {}
    return {}


def build_subculture_protocol(
    *,
    strain_id: str,
    strain_snapshot: dict[str, Any] | None,
    session_id: str | None = None,
    agent_run_id: str | None = None,
    source_message: str | None = None,
    parameters: dict[str, Any] | None = None,
    template: ProtocolTemplate | None = None,
    created_by: str = "protocol_builder",
    created_at: str | None = None,
) -> ExperimentProtocolSpec:
    if not strain_snapshot:
        raise ProtocolBuildError(
            "TARGET_STRAIN_NOT_FOUND",
            "Target strain was not found.",
            details={"strain_id": strain_id},
        )

    template = template or get_subculture_template()
    merged = {**DEFAULT_SUBCULTURE_PARAMETERS, **(parameters or {})}
    merged["expected_generation"] = int(strain_snapshot.get("generation_number") or 0)
    merged["strain_id"] = strain_id

    missing: list[str] = []
    for field_name in template.required_inputs:
        _required(merged.get(field_name), field_name, missing)
    if missing:
        raise ProtocolBuildError(
            "PROTOCOL_MISSING_REQUIRED_FIELD",
            "Protocol is missing required fields.",
            missing_fields=missing,
        )

    preconditions = [
        {
            "field": "generation_number",
            "equals": int(merged["expected_generation"]),
            "source": "algae_status",
        }
    ]
    steps = [
        ProtocolStep(
            step_id=template_step.step_id,
            operation=template_step.operation,
            parameters=_step_parameters(template_step.operation, merged),
            preconditions=preconditions if template_step.order == 1 else [],
            expected_observation={
                "template_step_order": template_step.order,
                "operation": template_step.operation,
            },
            failure_policy=template_step.failure_policy,
        )
        for template_step in sorted(template.steps, key=lambda item: item.order)
    ]
    now = created_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    protocol = ExperimentProtocolSpec(
        protocol_id="",
        protocol_version=1,
        template_id=template.template_id,
        template_version=template.template_version,
        experiment_type=template.experiment_type,
        objective=f"Execute controlled subculture for {strain_id}",
        target_strain_id=strain_id,
        inputs={
            "strain_snapshot": {
                "strain_id": strain_snapshot.get("strain_id"),
                "name_cn": strain_snapshot.get("name_cn"),
                "name_en": strain_snapshot.get("name_en"),
                "generation_number": int(strain_snapshot.get("generation_number") or 0),
                "days_since_last_subculture": int(
                    strain_snapshot.get("days_since_last_subculture") or 0
                ),
                "last_subculture_time": strain_snapshot.get("last_subculture_time"),
            },
            "source_message": source_message,
        },
        parameters=merged,
        steps=steps,
        hardware_requirements=list(template.required_hardware_capabilities),
        safety_constraints=[
            constraint.to_dict() for constraint in template.safety_constraints
        ],
        evidence_citations=[],
        expected_outputs=["workflow_run", "hardware_logs", "optional_db_commit"],
        created_at=now,
        created_by=created_by,
        source_run_id=agent_run_id,
        source_session_id=session_id,
    )
    protocol_hash = compute_protocol_hash(protocol)
    protocol.protocol_hash = protocol_hash
    protocol.protocol_id = f"protocol-{protocol_hash[:16]}"
    return protocol
