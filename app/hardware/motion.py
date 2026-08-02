from __future__ import annotations

from typing import Any, Iterable


MOTION_SCHEMA_VERSION = "motion-v2"


_RESOURCE_BY_ACTOR: dict[str, tuple[str, ...]] = {
    "robot_arm": ("fixed_robot_arm", "robot_gripper", "shared_transfer_zone"),
    "mobile_robot": ("mobile_robot", "mobile_robot_tray", "amr_route"),
    "human_operator": ("manual_handoff",),
    "liquid_handler": ("liquid_handler", "pipette_head", "liquid_handler_deck"),
    "media_pump": ("liquid_handler", "pipette_head", "media_channel", "target_flask_slot"),
    "seed_pump": ("liquid_handler", "pipette_head", "seed_channel", "target_flask_slot"),
    "plate_reader": ("plate_reader", "plate_reader_tray", "plate_reader_chamber"),
    "incubator": ("incubator", "incubator_door", "incubator_slot_1"),
    "safety_controller": ("safety_controller",),
}


def _phase(
    name: str,
    duration_ms: int,
    *,
    easing: str = "minimum_jerk",
    pose_from: str | None = None,
    pose_to: str | None = None,
    path: Iterable[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "duration_ms": int(duration_ms),
        "easing": easing,
    }
    if pose_from is not None:
        result["pose_from"] = pose_from
    if pose_to is not None:
        result["pose_to"] = pose_to
    if path:
        result["path"] = list(path)
    return result


def _profile(
    action: str,
    message: str,
    *,
    status: str = "running",
    progress: float | None = None,
) -> list[dict[str, Any]]:
    normalized = action.lower()
    text = message.lower()
    if normalized == "pick_labware":
        return [
            _phase("reserve_resources", 120, easing="hold"),
            _phase("approach", 380, pose_from="HOME", pose_to="PRE_PICK", path=("SAFE_Z", "PRE_PICK")),
            _phase("coarse_align", 180, pose_from="PRE_PICK", pose_to="PICK_APPROACH"),
            _phase("fine_align", 180, pose_from="PICK_APPROACH", pose_to="PICK"),
            _phase("grip_close", 160, easing="minimum_jerk"),
            _phase("grip_verify", 220, easing="hold"),
            _phase("lift", 440, pose_from="PICK", pose_to="CARRY_SAFE", path=("VERTICAL_CLEARANCE",)),
        ]
    if normalized == "transport_labware":
        return [
            _phase("accelerate", 500, easing="s_curve", pose_from="CARRY_SAFE", pose_to="TRANSFER_ENTRY"),
            _phase("cruise", 950, easing="linear_cruise", path=("TRANSFER_ENTRY", "SAFE_TRANSFER", "TARGET_ENTRY")),
            _phase("decelerate", 550, easing="s_curve", pose_from="TARGET_ENTRY", pose_to="PRE_PLACE"),
        ]
    if normalized == "place_labware":
        if status == "completed":
            return [_phase("placement_verified", 220, easing="hold")]
        return [
            _phase("target_align", 260, pose_from="PRE_PLACE", pose_to="PLACE_APPROACH"),
            _phase("lower", 320, pose_from="PLACE_APPROACH", pose_to="PLACE", path=("VERTICAL_APPROACH",)),
            _phase("seat_verify", 180, easing="hold"),
            _phase("release", 160),
            _phase("release_verify", 170, easing="hold"),
            _phase("retreat", 360, pose_from="PLACE", pose_to="CLEAR", path=("VERTICAL_CLEARANCE",)),
        ]
    if normalized in {"load_liquid_handler", "load_spectrophotometer", "store_measurement_plate", "move_to_incubator"}:
        return [_phase("return_home", 780, pose_from="CLEAR", pose_to="HOME", path=("SAFE_Z", "HOME"))]
    if normalized == "prepare_measurement_plate":
        if "aspirat" in text:
            return [
                _phase("pick_tip", 520),
                _phase("approach_source", 520, pose_from="TIP_RACK", pose_to="SOURCE_SAFE"),
                _phase("lower_to_liquid", 260, pose_from="SOURCE_SAFE", pose_to="SOURCE_LIQUID"),
                _phase("aspirate", 760, easing="s_curve"),
                _phase("aspirate_dwell", 260, easing="hold"),
                _phase("retract_vertical", 380, pose_from="SOURCE_LIQUID", pose_to="SOURCE_SAFE"),
            ]
        if "dispens" in text:
            return [
                _phase("move_above_well", 540, pose_from="SOURCE_SAFE", pose_to="WELL_SAFE"),
                _phase("lower_to_well", 240, pose_from="WELL_SAFE", pose_to="WELL_DISPENSE"),
                _phase("dispense", 660, easing="s_curve"),
                _phase("dispense_dwell", 240, easing="hold"),
                _phase("retract_vertical", 360, pose_from="WELL_DISPENSE", pose_to="WELL_SAFE"),
            ]
        return [
            _phase("prepare_blank", 820),
            _phase("dispense_blank", 760, easing="s_curve"),
            _phase("eject_tip", 420),
            _phase("return_home", 540, pose_to="HOME"),
        ]
    if normalized in {"open_door", "close_door"}:
        return [
            _phase("interlock_check", 200, easing="hold"),
            _phase("door_motion", 600, easing="minimum_jerk"),
            _phase("door_latch_verify", 220, easing="hold"),
        ]
    if normalized == "blank_plate":
        if status == "completed":
            return [_phase("blank_verify", 350, easing="hold")]
        return [
            _phase("tray_retract", 520, easing="minimum_jerk"),
            _phase("optics_settle", 560, easing="hold"),
            _phase("blank_read", 1460, easing="hold"),
            _phase("blank_verify", 400, easing="hold"),
        ]
    if normalized == "measure_plate_absorbance":
        if status == "completed":
            return [_phase("result_validate", 350, easing="hold")]
        return [
            _phase("select_wavelength", 350, easing="hold"),
            _phase("sample_read", 1540, easing="hold"),
            _phase("result_validate", 410, easing="hold"),
        ]
    if normalized == "dispense_medium":
        if status == "completed":
            return [
                _phase("close_valve", 240),
                _phase("drip_dwell", 280, easing="hold"),
                _phase("retract_vertical", 460, pose_from="TARGET_DISPENSE", pose_to="TARGET_SAFE"),
            ]
        if progress is not None and progress > 0.1:
            return [_phase("metered_dispense", 360, easing="s_curve")]
        return [
            _phase("interlock_check", 260, easing="hold"),
            _phase("approach_target", 620, pose_from="HOME", pose_to="TARGET_SAFE"),
            _phase("lower_to_dispense", 320, pose_from="TARGET_SAFE", pose_to="TARGET_DISPENSE"),
            _phase("open_valve", 240),
            _phase("metered_dispense", 360, easing="s_curve"),
        ]
    if normalized == "transfer_seed":
        if status == "completed":
            return [_phase("dispense_dwell", 260, easing="hold"), _phase("return_home", 520, pose_to="HOME")]
        if progress is not None and progress > 0.125:
            return [_phase("dispense_seed", 320, easing="s_curve")]
        return [
            _phase("approach_source", 520, pose_from="HOME", pose_to="SOURCE_SAFE"),
            _phase("aspirate_seed", 720, easing="s_curve"),
            _phase("aspirate_dwell", 240, easing="hold"),
            _phase("move_to_target", 620, pose_from="SOURCE_SAFE", pose_to="TARGET_SAFE"),
            _phase("dispense_seed", 320, easing="s_curve"),
        ]
    if normalized == "mix_and_seal":
        if status == "completed":
            return [_phase("seal_verify", 400, easing="hold")]
        return [
            _phase("mix", 1260, easing="s_curve"),
            _phase("settle", 320, easing="hold"),
            _phase("seal", 620),
            _phase("seal_verify", 280, easing="hold"),
        ]
    if normalized == "configure":
        return [
            _phase("door_open", 720),
            _phase("slot_verify", 320, easing="hold"),
            _phase("door_close", 720),
            _phase("set_environment", 760, easing="hold"),
        ]
    if normalized == "safe_shutdown":
        return [
            _phase("controlled_deceleration", 520, easing="s_curve"),
            _phase("stop_flow", 180, easing="hold"),
            _phase("close_valves", 260),
            _phase("retract_vertical", 620, pose_from="PROCESS", pose_to="SAFE_Z"),
            _phase("return_safe_home", 920, pose_from="SAFE_Z", pose_to="HOME"),
            _phase("release_resources", 240, easing="hold"),
        ]
    if normalized == "return_home":
        return [
            _phase("retract_vertical", 420, pose_from="PROCESS", pose_to="SAFE_Z"),
            _phase("return_safe_home", 720, pose_from="SAFE_Z", pose_to="HOME", path=("SAFE_Z", "HOME")),
            _phase("home_verify", 220, easing="hold"),
        ]
    if normalized == "cleanup":
        return [_phase("clear_deck", 820), _phase("waste_confirm", 360, easing="hold"), _phase("return_home", 520)]
    if normalized in {"check", "sterilize", "close_reactor"}:
        return [_phase("interlock_check", 480, easing="hold"), _phase("operation", 920), _phase("verify", 360, easing="hold")]
    return [_phase("state_transition", 420, easing="hold")]


def build_motion_command(
    *,
    sequence: int,
    actor_id: str,
    entity_id: str | None,
    action_type: str,
    message: str,
    simulation_start_ms: int,
    source_location: str | None = None,
    target_location: str | None = None,
    attachment_transition: dict[str, Any] | None = None,
    additional_resources: Iterable[str] = (),
    status: str = "running",
    progress: float | None = None,
) -> dict[str, Any]:
    phases = _profile(action_type, message, status=status, progress=progress)
    resources = list(dict.fromkeys((*_RESOURCE_BY_ACTOR.get(actor_id, (actor_id,)), *additional_resources)))
    command_id = f"motion-{sequence:05d}"
    command: dict[str, Any] = {
        "schema_version": MOTION_SCHEMA_VERSION,
        "command_id": command_id,
        "actor_id": actor_id,
        "entity_id": entity_id,
        "action_type": action_type,
        "simulation_start_ms": int(simulation_start_ms),
        "nominal_duration_ms": sum(int(phase["duration_ms"]) for phase in phases),
        "required_resources": resources,
        "preconditions": [f"resource:{resource}:available" for resource in resources],
        "postconditions": [f"action:{action_type}:completed"],
        "phases": phases,
        "interruptibility": "phase_boundary",
        "safe_stop_phase": "retract_vertical" if action_type in {"dispense_medium", "transfer_seed", "safe_shutdown"} else "hold_position",
    }
    if source_location is not None:
        command["source_location"] = source_location
    if target_location is not None:
        command["target_location"] = target_location
    if attachment_transition:
        command["attachment_transition"] = attachment_transition
    return command
