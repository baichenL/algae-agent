# 从数据库读取目标品系状态，构造 LangGraph 初始 state
# 调用 LangGraph 执行物理操作流程
# 从 LangGraph 输出中提取结构化工具结果；数据库提交由 WorkflowResultCommitService 负责
import asyncio
import os
from typing import Any, Callable, Dict

from app.agent.graph import build_subculture_workflow
from app.core.database import get_algae_status
from app.hardware import SimulatedHardware


async def run_force_subculture_workflow(
    strain: str,
    execution_mode: str | None = None,
    compiled_protocol: dict[str, Any] | None = None,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
    delay_seconds: float | None = None,
) -> Dict[str, Any]:
    status_data = get_algae_status(strain)
    if not status_data:
        return {
            "status": "error",
            "message": "Strain record was not found in the database",
            "strain_id": strain,
            "execution_logs": [],
        }

    mode = (execution_mode or os.getenv("HARDWARE_MODE", "simulation")).strip().lower()
    if mode != "simulation":
        return {
            "status": "error",
            "message": "Real hardware adapter is not configured; production execution is blocked",
            "strain_id": strain,
            "execution_mode": mode,
            "execution_logs": [],
        }

    if delay_seconds is None:
        is_test = bool(os.getenv("PYTEST_CURRENT_TEST")) or os.getenv("ALGAE_AUTH_MODE") == "test"
        configured_delay = os.getenv("SUBCULTURE_SIMULATION_DELAY")
        delay_seconds = float(configured_delay) if configured_delay is not None else (0.0 if is_test else 0.28)
    hardware = SimulatedHardware(delay_seconds=delay_seconds, event_sink=event_sink)
    workflow = build_subculture_workflow(hardware)

    if compiled_protocol:
        forced_state = dict(compiled_protocol.get("initial_state") or {})
        forced_state["hardware_state"] = hardware.snapshot()
        forced_state["execution_mode"] = "simulation"
    else:
        forced_state = {
            "messages": [],
            "execution_mode": "simulation",
            "strain_id": strain,
            "generation_number": status_data.get("generation_number", 1),
            "inoculation_timestamp": "",
            "source_reactor_id": "Reactor_A",
            "target_reactor_id": "",
            "media_target_volume": 150.0,
            "seed_target_volume": 1.0,
            "days_since_last_subculture": 6,
            "current_step": "START",
            "hardware_logs": [],
            "simulation_events": [],
            "hardware_state": hardware.snapshot(),
            "last_error": None,
            "workflow_status": "IDLE",
        }

    final_state = await asyncio.to_thread(workflow.invoke, forced_state)
    workflow_status = final_state.get("workflow_status", "FAILED")
    if workflow_status != "SUCCESS":
        return {
            "status": "error",
            "message": "Simulated subculture workflow failed",
            "strain_id": strain,
            "execution_mode": "simulation",
            "execution_logs": final_state.get("hardware_logs", []),
            "hardware_state": final_state.get("hardware_state", hardware.snapshot()),
            "last_error": final_state.get("last_error"),
            "persisted": False,
        }

    return {
        "status": "success",
        "message": f"Completed simulated subculture for generation {final_state.get('generation_number')}",
        "execution_mode": "simulation",
        "physical_execution": False,
        "persisted": False,
        "execution_logs": final_state.get("hardware_logs", []),
        "hardware_state": final_state.get("hardware_state", hardware.snapshot()),
        "simulation_events": final_state.get("simulation_events", []),
        "current_generation": final_state.get("generation_number"),
        "days_counter": final_state.get("days_since_last_subculture"),
        "inoculation_time": final_state.get("inoculation_timestamp"),
        "protocol_id": final_state.get("protocol_id"),
        "protocol_hash": final_state.get("protocol_hash"),
    }
