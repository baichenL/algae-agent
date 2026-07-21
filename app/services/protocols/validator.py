from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.hardware import HardwareController, SimulatedHardware
from app.services.labware import (
    get_container_ref,
    get_labware_spec,
    get_workspace_location,
)
from app.services.protocols.hashing import verify_protocol_hash
from app.services.protocols.models import (
    ExperimentProtocolSpec,
    ProtocolIssue,
    ProtocolTemplate,
    ProtocolValidationCheck,
    ProtocolValidationReport,
)
from app.services.protocols.templates import (
    OPERATION_TO_HAL_CAPABILITY,
    SUPPORTED_PROTOCOL_OPERATIONS,
    get_subculture_template,
)


VALIDATOR_VERSION = "protocol_validator_v1"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _issue(
    code: str,
    message: str,
    *,
    step_id: str | None = None,
    field_path: str | None = None,
    details: dict[str, Any] | None = None,
    level: str = "error",
) -> ProtocolIssue:
    return ProtocolIssue(
        code=code,
        level=level,
        message=message,
        step_id=step_id,
        field_path=field_path,
        details=details or {},
    )


def _check(
    checks: list[ProtocolValidationCheck],
    code: str,
    passed: bool,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    checks.append(
        ProtocolValidationCheck(
            check_code=code,
            passed=passed,
            message=message,
            details=details or {},
        )
    )


def _hardware_capabilities(controller: HardwareController | None = None) -> set[str]:
    hardware = controller or SimulatedHardware(delay_seconds=0)
    return {
        name
        for name in OPERATION_TO_HAL_CAPABILITY.values()
        if callable(getattr(hardware, name, None))
    }


def _parameter_valid(value: Any, value_type: str) -> bool:
    if value_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if value_type == "string":
        return isinstance(value, str) and bool(value)
    return True


def _duplicate_pending_exists(
    pending_actions: list[dict[str, Any]],
    protocol: ExperimentProtocolSpec,
) -> bool:
    for item in pending_actions or []:
        if item.get("status") != "pending":
            continue
        payload = item.get("payload") or {}
        data = payload.get("data") or {}
        frozen = payload.get("protocol") or {}
        if payload.get("type") != "workflow_subculture":
            continue
        if data.get("strain_id") != protocol.target_strain_id:
            continue
        if frozen.get("protocol_hash") in {None, protocol.protocol_hash}:
            return True
    return False


def _validate_labware_workspace(
    protocol: ExperimentProtocolSpec,
    checks: list[ProtocolValidationCheck],
    errors: list[ProtocolIssue],
) -> None:
    source_id = protocol.parameters.get("source_reactor_id")
    target_id = protocol.parameters.get("target_reactor_id")
    container_ids = [item for item in [source_id, target_id] if item]
    missing_containers: list[str] = []
    missing_labware: list[str] = []
    missing_locations: list[str] = []

    for container_id in container_ids:
        container = get_container_ref(str(container_id))
        if container is None:
            missing_containers.append(str(container_id))
            continue
        if get_labware_spec(container.labware_id) is None:
            missing_labware.append(container.labware_id)
        if get_workspace_location(container.location_id) is None:
            missing_locations.append(container.location_id)

    registry_ok = not missing_containers and not missing_labware and not missing_locations
    _check(
        checks,
        "LABWARE_WORKSPACE_REGISTERED",
        registry_ok,
        "Protocol containers, labware, and workspace locations are registered.",
        {
            "missing_containers": missing_containers,
            "missing_labware": missing_labware,
            "missing_locations": missing_locations,
        },
    )
    if not registry_ok:
        errors.append(
            _issue(
                "LABWARE_WORKSPACE_UNKNOWN",
                "Protocol references unknown labware or workspace locations.",
                details={
                    "missing_containers": missing_containers,
                    "missing_labware": missing_labware,
                    "missing_locations": missing_locations,
                },
            )
        )
        return

    target = get_container_ref(str(target_id)) if target_id else None
    target_spec = get_labware_spec(target.labware_id) if target else None
    planned_target_volume = float(protocol.parameters.get("media_target_volume") or 0) + float(
        protocol.parameters.get("seed_target_volume") or 0
    )
    capacity_ok = (
        target_spec is None
        or target_spec.max_volume_ml is None
        or planned_target_volume <= float(target_spec.max_volume_ml)
    )
    _check(
        checks,
        "LABWARE_CAPACITY_AVAILABLE",
        capacity_ok,
        "Target labware has enough nominal capacity for planned volume.",
        {
            "target_container_id": target_id,
            "planned_volume_ml": planned_target_volume,
            "max_volume_ml": target_spec.max_volume_ml if target_spec else None,
        },
    )
    if not capacity_ok:
        errors.append(
            _issue(
                "LABWARE_CAPACITY_EXCEEDED",
                "Planned protocol volume exceeds target labware capacity.",
                field_path="parameters.media_target_volume",
                details={
                    "target_container_id": target_id,
                    "planned_volume_ml": planned_target_volume,
                    "max_volume_ml": target_spec.max_volume_ml if target_spec else None,
                },
            )
        )

    unsupported_locations = []
    for step in protocol.steps:
        location_id = None
        if step.operation in {
            "sterilize_workspace",
            "check_materials",
            "dispense_medium",
            "transfer_seed",
            "close_reactor",
            "cleanup",
        }:
            location_id = "biosafety_cabinet"
        elif step.operation == "configure_incubator":
            location_id = "incubator_1"
        if not location_id:
            continue
        location = get_workspace_location(location_id)
        if location and step.operation not in location.supported_operations:
            unsupported_locations.append({
                "step_id": step.step_id,
                "operation": step.operation,
                "location_id": location_id,
            })
    location_ok = not unsupported_locations
    _check(
        checks,
        "WORKSPACE_OPERATION_SUPPORTED",
        location_ok,
        "Workspace locations support all planned operations.",
        {"unsupported": unsupported_locations},
    )
    if not location_ok:
        errors.append(
            _issue(
                "WORKSPACE_OPERATION_UNSUPPORTED",
                "A protocol operation is not supported at the assigned workspace location.",
                details={"unsupported": unsupported_locations},
            )
        )


def validate_protocol(
    protocol: ExperimentProtocolSpec,
    *,
    strain_snapshot: dict[str, Any] | None,
    all_strains: list[dict[str, Any]] | None = None,
    pending_actions: list[dict[str, Any]] | None = None,
    template: ProtocolTemplate | None = None,
    hardware_controller: HardwareController | None = None,
    enforce_duplicate_pending: bool = True,
) -> ProtocolValidationReport:
    template = template or get_subculture_template()
    checks: list[ProtocolValidationCheck] = []
    errors: list[ProtocolIssue] = []
    warnings: list[ProtocolIssue] = []

    target_matches = [
        item
        for item in all_strains or []
        if item.get("strain_id") == protocol.target_strain_id
    ]
    strain_exists = bool(strain_snapshot)
    if not all_strains:
        target_matches = [strain_snapshot] if strain_snapshot else []
    _check(checks, "TARGET_STRAIN_EXISTS", strain_exists, "Target strain exists.")
    if not strain_exists:
        errors.append(
            _issue(
                "TARGET_STRAIN_NOT_FOUND",
                "Target strain was not found.",
                field_path="target_strain_id",
                details={"target_strain_id": protocol.target_strain_id},
            )
        )

    unique = len(target_matches) <= 1
    _check(checks, "TARGET_STRAIN_UNIQUE", unique, "Target strain is unique.")
    if not unique:
        errors.append(
            _issue(
                "TARGET_STRAIN_AMBIGUOUS",
                "Multiple target strain records matched the protocol target.",
                field_path="target_strain_id",
                details={"target_strain_id": protocol.target_strain_id},
            )
        )

    generation_ok = True
    expected_generation = protocol.parameters.get("expected_generation")
    current_generation = None
    if strain_snapshot:
        current_generation = int(strain_snapshot.get("generation_number") or 0)
        generation_ok = (
            expected_generation is None
            or int(expected_generation) == current_generation
        )
    _check(
        checks,
        "TARGET_GENERATION_CURRENT",
        generation_ok,
        "Protocol generation matches database state.",
        {
            "expected_generation": expected_generation,
            "current_generation": current_generation,
        },
    )
    if not generation_ok:
        errors.append(
            _issue(
                "TARGET_STATE_STALE",
                "Protocol expected generation does not match current database state.",
                field_path="parameters.expected_generation",
                details={
                    "expected_generation": expected_generation,
                    "current_generation": current_generation,
                },
            )
        )

    duplicate = enforce_duplicate_pending and _duplicate_pending_exists(
        pending_actions or [],
        protocol,
    )
    _check(checks, "NO_DUPLICATE_PENDING", not duplicate, "No duplicate pending exists.")
    if duplicate:
        errors.append(
            _issue(
                "DUPLICATE_PENDING",
                "An active pending subculture request already exists for this target.",
                details={"target_strain_id": protocol.target_strain_id},
            )
        )

    missing = [
        field_name
        for field_name in template.required_inputs
        if protocol.parameters.get(field_name) in {None, ""}
    ]
    _check(
        checks,
        "REQUIRED_PARAMETERS_PRESENT",
        not missing,
        "Required protocol parameters are present.",
        {"missing_fields": missing},
    )
    for field_name in missing:
        errors.append(
            _issue(
                "PROTOCOL_MISSING_REQUIRED_FIELD",
                "Protocol parameter is missing.",
                field_path=f"parameters.{field_name}",
            )
        )

    for field_name, allowed_range in template.allowed_parameter_ranges.items():
        if field_name not in protocol.parameters:
            continue
        value = protocol.parameters.get(field_name)
        type_ok = _parameter_valid(value, allowed_range.value_type)
        range_ok = type_ok
        if type_ok and isinstance(value, int | float):
            if allowed_range.min_value is not None:
                range_ok = range_ok and float(value) >= float(allowed_range.min_value)
            if allowed_range.max_value is not None:
                range_ok = range_ok and float(value) <= float(allowed_range.max_value)
        _check(
            checks,
            f"PARAMETER_VALID:{field_name}",
            type_ok and range_ok,
            "Protocol parameter type and range are valid.",
            {"field": field_name, "value": value, "range": allowed_range.to_dict()},
        )
        if not type_ok or not range_ok:
            errors.append(
                _issue(
                    "PROTOCOL_INVALID_PARAMETER",
                    "Protocol parameter type or range is invalid.",
                    field_path=f"parameters.{field_name}",
                    details={"value": value, "range": allowed_range.to_dict()},
                )
            )

    template_steps = sorted(template.steps, key=lambda item: item.order)
    expected_step_ids = [step.step_id for step in template_steps if step.required]
    actual_step_ids = [step.step_id for step in protocol.steps]
    steps_complete = expected_step_ids == actual_step_ids
    _check(
        checks,
        "PROTOCOL_STEPS_COMPLETE",
        steps_complete,
        "Protocol contains the fixed required template steps.",
        {"expected": expected_step_ids, "actual": actual_step_ids},
    )
    if not steps_complete:
        errors.append(
            _issue(
                "PROTOCOL_STEP_ORDER_INVALID",
                "Protocol steps are missing or out of order.",
                details={"expected": expected_step_ids, "actual": actual_step_ids},
            )
        )

    template_order_by_step = {step.step_id: step.order for step in template_steps}
    actual_orders = [template_order_by_step.get(step.step_id, -1) for step in protocol.steps]
    order_ok = actual_orders == sorted(actual_orders) and -1 not in actual_orders
    _check(
        checks,
        "PROTOCOL_STEP_ORDER_VALID",
        order_ok,
        "Protocol step order is valid.",
        {"actual_orders": actual_orders},
    )
    if not order_ok and steps_complete:
        errors.append(
            _issue(
                "PROTOCOL_STEP_ORDER_INVALID",
                "Protocol step order is invalid.",
                details={"actual_orders": actual_orders},
            )
        )

    for step in protocol.steps:
        operation_ok = step.operation in SUPPORTED_PROTOCOL_OPERATIONS
        _check(
            checks,
            f"OPERATION_ALLOWED:{step.step_id}",
            operation_ok,
            "Protocol operation is allow-listed.",
            {"operation": step.operation},
        )
        if not operation_ok:
            errors.append(
                _issue(
                    "PROTOCOL_OPERATION_NOT_ALLOWED",
                    "Protocol operation is not allow-listed.",
                    step_id=step.step_id,
                    field_path="operation",
                    details={"operation": step.operation},
                )
            )

    capabilities = _hardware_capabilities(hardware_controller)
    missing_capabilities = sorted(set(protocol.hardware_requirements) - capabilities)
    hardware_ok = not missing_capabilities
    _check(
        checks,
        "HARDWARE_CAPABILITIES_AVAILABLE",
        hardware_ok,
        "HAL supports required protocol capabilities.",
        {"missing": missing_capabilities, "available": sorted(capabilities)},
    )
    if not hardware_ok:
        errors.append(
            _issue(
                "HARDWARE_CAPABILITY_MISSING",
                "HAL does not support required protocol capabilities.",
                details={"missing": missing_capabilities},
            )
        )

    _validate_labware_workspace(protocol, checks, errors)

    approval_ok = template.requires_approval is True
    _check(
        checks,
        "APPROVAL_REQUIRED",
        approval_ok,
        "Protocol template requires approval.",
        {"template_id": template.template_id},
    )
    if not approval_ok:
        errors.append(
            _issue(
                "APPROVAL_REQUIRED",
                "Protocol template must require approval.",
                field_path="template.requires_approval",
            )
        )

    hash_ok = verify_protocol_hash(protocol)
    _check(
        checks,
        "PROTOCOL_HASH_VALID",
        hash_ok,
        "Protocol hash matches canonical protocol content.",
        {"protocol_hash": protocol.protocol_hash},
    )
    if not hash_ok:
        errors.append(
            _issue(
                "PROTOCOL_HASH_MISMATCH",
                "Protocol hash does not match canonical protocol content.",
                field_path="protocol_hash",
                details={"protocol_hash": protocol.protocol_hash},
            )
        )

    route_ok = protocol.inputs.get("source_route_kind") not in {"rag", "chat"}
    _check(
        checks,
        "CONTROLLED_PROTOCOL_ROUTE",
        route_ok,
        "Protocol execution is not sourced from an uncontrolled chat or RAG route.",
        {"source_route_kind": protocol.inputs.get("source_route_kind")},
    )
    if not route_ok:
        errors.append(
            _issue(
                "PROTOCOL_OPERATION_NOT_ALLOWED",
                "Protocol cannot be executed directly from RAG or generic chat route.",
                field_path="inputs.source_route_kind",
            )
        )

    return ProtocolValidationReport(
        protocol_id=protocol.protocol_id,
        valid=not errors,
        checks=checks,
        errors=errors,
        warnings=warnings,
        validated_at=_now(),
        validator_version=VALIDATOR_VERSION,
    )
