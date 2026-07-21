from __future__ import annotations

from datetime import datetime
from typing import Any

from langgraph.types import interrupt

from app.agent.state import AlgaeSubcultureState
from app.hardware import HardwareController, HardwareResult, SimulatedHardware


_default_hardware = SimulatedHardware(delay_seconds=0.0)


def _hardware(controller: HardwareController | None) -> HardwareController:
    return controller or _default_hardware


def _apply_hardware_result(
    state: AlgaeSubcultureState,
    *,
    step: str,
    result: HardwareResult,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "current_step": step,
        "hardware_state": result.state,
        "simulation_events": list(state.get("simulation_events", [])) + result.events,
        "hardware_logs": list(state.get("hardware_logs", [])) + [result.message],
    }
    if result.ok:
        update["last_error"] = None
    else:
        update.update(
            {
                "workflow_status": "FAILED",
                "last_error": {
                    "step": step,
                    "device": result.device,
                    "code": result.error_code or "HARDWARE_FAILURE",
                    "message": result.message,
                },
            }
        )
    if extra:
        update.update(extra)
    return update


def check_schedule_node(state: AlgaeSubcultureState) -> dict[str, Any]:
    days = int(state.get("days_since_last_subculture", 0))
    current_gen = int(state.get("generation_number", 1))
    logs = list(state.get("hardware_logs", []))
    if days < 6:
        return {
            "current_step": "CheckSchedule",
            "workflow_status": "SKIPPED",
            "hardware_logs": logs + [
                f"Cycle is {days} days; the 6-day subculture threshold is not met."
            ],
        }
    return {
        "current_step": "CheckSchedule",
        "workflow_status": "RUNNING",
        "target_reactor_id": state.get("target_reactor_id") or "Reactor_B",
        "hardware_logs": logs
        + [f"Starting simulated subculture F{current_gen} -> F{current_gen + 1}."],
    }


def _manual_transfer_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None,
    task_id: str,
    step: str,
    title: str,
    source: str,
    destination: str,
    container: str,
    instruction: str,
) -> dict[str, Any]:
    task = {
        "task_id": task_id,
        "step": step,
        "title": title,
        "source": source,
        "destination": destination,
        "container": container,
        "instruction": instruction,
    }
    mode = state.get("interaction_mode", "legacy")
    resolution = (
        {"resolution": "completed", "note": "legacy auto-complete"}
        if mode not in {"verification", "demo"}
        else interrupt(task)
    )
    if resolution.get("resolution") != "completed":
        message = resolution.get("note") or f"Manual task failed: {title}"
        return {
            "current_step": step,
            "workflow_status": "FAILED",
            "hardware_state": _hardware(controller).snapshot(),
            "hardware_logs": list(state.get("hardware_logs", [])) + [message],
            "last_error": {
                "step": step,
                "device": "human_operator",
                "code": "MANUAL_TASK_FAILED",
                "message": message,
            },
            "pending_manual_task": None,
        }
    result = _hardware(controller).relocate_sample(
        task_id=task_id,
        location=destination,
        container=container,
    )
    return _apply_hardware_result(
        state,
        step=step,
        result=result,
        extra={"pending_manual_task": None},
    )


def manual_load_spectrophotometer_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    return _manual_transfer_node(
        state,
        controller=controller,
        task_id="load_spectrophotometer",
        step="ManualLoadSpectrophotometer",
        title="将培养样品放入分光光度计",
        source="人工操作区",
        destination="spectrophotometer",
        container="source_flask",
        instruction="取源三角烧瓶样品，装入比色皿并放入分光光度计。",
    )


def blank_spectrophotometer_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).spectrophotometer.blank(
        float(state.get("wavelength_nm", 680.0))
    )
    return _apply_hardware_result(
        state,
        step="BlankSpectrophotometer",
        result=result,
    )


def measure_absorbance_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).spectrophotometer.measure_absorbance(
        float(state.get("wavelength_nm", 680.0)),
        float(state.get("simulated_absorbance", 0.8)),
    )
    measured = (result.state.get("spectrophotometer") or {}).get("absorbance")
    return _apply_hardware_result(
        state,
        step="MeasureAbsorbance",
        result=result,
        extra={"measured_absorbance": measured} if result.ok else None,
    )


def manual_load_liquid_handler_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    return _manual_transfer_node(
        state,
        controller=controller,
        task_id="load_liquid_handler",
        step="ManualLoadLiquidHandler",
        title="装载移液平台",
        source="人工操作区",
        destination="liquid_handler",
        container="source_and_target_flasks",
        instruction="放置源三角烧瓶、目标三角烧瓶、培养基和所需耗材。",
    )


def sterilize_workspace_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).sterilize_workspace(duration_min=30)
    return _apply_hardware_result(state, step="SterilizeWorkspace", result=result)


def load_materials_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).liquid_handler.check_materials(
        expected_media_ml=float(state["media_target_volume"])
    )
    return _apply_hardware_result(state, step="LoadMaterials", result=result)


def dispense_medium_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).liquid_handler.dispense_medium(
        target=state["target_reactor_id"],
        volume_ml=float(state["media_target_volume"]),
    )
    return _apply_hardware_result(state, step="DispenseMedium", result=result)


def transfer_seed_culture_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).liquid_handler.transfer_seed(
        source=state["source_reactor_id"],
        target=state["target_reactor_id"],
        volume_ml=float(state["seed_target_volume"]),
    )
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    extra = {"inoculation_timestamp": timestamp} if result.ok else None
    return _apply_hardware_result(
        state,
        step="TransferSeedCulture",
        result=result,
        extra=extra,
    )


def seal_culture_bottle_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).liquid_handler.mix_and_seal(
        state["target_reactor_id"]
    )
    return _apply_hardware_result(state, step="MixAndSeal", result=result)


def manual_move_to_incubator_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    return _manual_transfer_node(
        state,
        controller=controller,
        task_id="move_to_incubator",
        step="ManualMoveToIncubator",
        title="将目标培养瓶移入培养箱",
        source="liquid_handler",
        destination="incubator",
        container="target_flask",
        instruction="取下已封口的目标三角烧瓶，并放入指定培养箱位置。",
    )


def move_to_incubator_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).incubator_controller.configure(
        temperature_c=25.0,
        light_lux=3500.0,
    )
    return _apply_hardware_result(state, step="MoveToIncubator", result=result)


def record_experiment_node(state: AlgaeSubcultureState) -> dict[str, Any]:
    old_generation = int(state.get("generation_number", 1))
    return {
        "current_step": "RecordExperiment",
        "generation_number": old_generation + 1,
        "days_since_last_subculture": 0,
        "hardware_logs": list(state.get("hardware_logs", []))
        + [f"Simulation result prepared: F{old_generation} -> F{old_generation + 1}."],
    }


def cleanup_workspace_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    result = _hardware(controller).cleanup()
    update = _apply_hardware_result(state, step="CleanupWorkspace", result=result)
    update["workflow_status"] = "SUCCESS" if result.ok else "FAILED"
    return update


def safe_shutdown_node(
    state: AlgaeSubcultureState,
    *,
    controller: HardwareController | None = None,
) -> dict[str, Any]:
    error = state.get("last_error") or {}
    reason = error.get("message") or "workflow failure"
    result = _hardware(controller).safe_shutdown(reason)
    return {
        "current_step": "SafeShutdown",
        "workflow_status": "FAILED",
        "hardware_state": result.state,
        "simulation_events": list(state.get("simulation_events", [])) + result.events,
        "hardware_logs": list(state.get("hardware_logs", []))
        + [result.message, f"Workflow stopped safely: {reason}"],
        "last_error": error,
    }
