from __future__ import annotations

from app.services.protocols.models import (
    ParameterRange,
    ProtocolConstraint,
    ProtocolTemplate,
    ProtocolTemplateStep,
)


SUBCULTURE_EXPERIMENT_TYPE = "subculture"

LOGICAL_OPERATIONS = {
    "check_schedule",
    "record_experiment",
}

HARDWARE_OPERATIONS = {
    "sterilize_workspace",
    "check_materials",
    "dispense_medium",
    "transfer_seed",
    "close_reactor",
    "configure_incubator",
    "cleanup",
}

SUPPORTED_PROTOCOL_OPERATIONS = LOGICAL_OPERATIONS | HARDWARE_OPERATIONS

OPERATION_TO_HAL_CAPABILITY = {
    "sterilize_workspace": "sterilize_workspace",
    "check_materials": "check_materials",
    "dispense_medium": "dispense_medium",
    "transfer_seed": "transfer_seed",
    "close_reactor": "close_reactor",
    "configure_incubator": "configure_incubator",
    "cleanup": "cleanup",
}

DEFAULT_SUBCULTURE_PARAMETERS = {
    "days_since_last_subculture": 6,
    "source_reactor_id": "Reactor_A",
    "target_reactor_id": "Reactor_B",
    "media_target_volume": 150.0,
    "seed_target_volume": 1.0,
    "temperature_c": 25.0,
    "light_lux": 3500.0,
}


SUBCULTURE_TEMPLATE = ProtocolTemplate(
    template_id="subculture_standard_v1",
    template_version=1,
    experiment_type=SUBCULTURE_EXPERIMENT_TYPE,
    name="Standard subculture workflow",
    description="Fixed protocol skeleton for the existing algae subculture workflow.",
    required_inputs=[
        "strain_id",
        "expected_generation",
        "source_reactor_id",
        "target_reactor_id",
        "media_target_volume",
        "seed_target_volume",
    ],
    required_hardware_capabilities=sorted(OPERATION_TO_HAL_CAPABILITY.values()),
    steps=[
        ProtocolTemplateStep("check_schedule", "check_schedule", True, 1),
        ProtocolTemplateStep("sterilize_workspace", "sterilize_workspace", True, 2),
        ProtocolTemplateStep("load_materials", "check_materials", True, 3),
        ProtocolTemplateStep("dispense_medium", "dispense_medium", True, 4),
        ProtocolTemplateStep("transfer_seed_culture", "transfer_seed", True, 5),
        ProtocolTemplateStep("seal_culture_bottle", "close_reactor", True, 6),
        ProtocolTemplateStep("move_to_incubator", "configure_incubator", True, 7),
        ProtocolTemplateStep("record_experiment", "record_experiment", True, 8),
        ProtocolTemplateStep("cleanup_workspace", "cleanup", True, 9),
    ],
    safety_constraints=[
        ProtocolConstraint(
            code="APPROVAL_REQUIRED",
            description="Subculture workflow must stop at pending approval before execution.",
        ),
        ProtocolConstraint(
            code="SAFE_SHUTDOWN_ON_HARDWARE_FAILURE",
            description="Any hardware failure must stop later steps and place devices in a safe state.",
        ),
    ],
    allowed_parameter_ranges={
        "days_since_last_subculture": ParameterRange(0, 365, "integer"),
        "expected_generation": ParameterRange(1, 100000, "integer"),
        "media_target_volume": ParameterRange(1.0, 240.0, "number"),
        "seed_target_volume": ParameterRange(0.1, 50.0, "number"),
        "temperature_c": ParameterRange(1.0, 40.0, "number"),
        "light_lux": ParameterRange(0.0, 20000.0, "number"),
    },
    requires_approval=True,
    active=True,
)


def get_subculture_template() -> ProtocolTemplate:
    return SUBCULTURE_TEMPLATE
