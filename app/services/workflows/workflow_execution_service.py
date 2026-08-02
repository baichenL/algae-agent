from __future__ import annotations

import inspect
import threading
from typing import Any, Callable

from app.core import database
from app.core.db import control_plane as control_plane_db
from app.core.db import workflow_runs
from app.services.agent_runtime import record_run_event
from app.services.memory.memory_service import append_decision_event
from app.services.observability.error_events import record_error_event
from app.services.protocols import compile_protocol_to_workflow
from app.services.protocols.pending_payload import (
    load_frozen_protocol,
    verify_frozen_protocol_payload,
)
from app.services.workflows.workflow_result_commit_service import commit_workflow_result
from app.services.workflows.workflow_service import run_force_subculture_workflow


async def _run_subculture_workflow(
    strain_id: str,
    *,
    execution_mode: str | None,
    compiled_protocol: dict | None,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    signature = inspect.signature(run_force_subculture_workflow)
    kwargs: dict[str, Any] = {"execution_mode": execution_mode}
    if "compiled_protocol" in signature.parameters:
        kwargs["compiled_protocol"] = compiled_protocol
    if "event_sink" in signature.parameters:
        kwargs["event_sink"] = event_sink
    return await run_force_subculture_workflow(strain_id, **kwargs)


_SIMULATION_STEPS = [
    "CheckSchedule",
    "ManualLoadLiquidHandler",
    "PrepareMeasurementPlate",
    "ManualLoadSpectrophotometer",
    "BlankSpectrophotometer",
    "MeasureAbsorbance",
    "StoreMeasurementPlate",
    "LoadMaterials",
    "DispenseMedium",
    "TransferSeedCulture",
    "MixAndSeal",
    "ManualMoveToIncubator",
    "MoveToIncubator",
    "RecordExperiment",
    "CleanupWorkspace",
]

_ACTION_TO_STEP = {
    "load_spectrophotometer": "ManualLoadSpectrophotometer",
    "blank": "BlankSpectrophotometer",
    "blank_plate": "BlankSpectrophotometer",
    "measure_absorbance": "MeasureAbsorbance",
    "measure_plate_absorbance": "MeasureAbsorbance",
    "load_liquid_handler": "ManualLoadLiquidHandler",
    "prepare_measurement_plate": "PrepareMeasurementPlate",
    "store_measurement_plate": "StoreMeasurementPlate",
    "check": "LoadMaterials",
    "dispense_medium": "DispenseMedium",
    "transfer_seed": "TransferSeedCulture",
    "mix_and_seal": "MixAndSeal",
    "move_to_incubator": "ManualMoveToIncubator",
    "configure": "MoveToIncubator",
    "cleanup": "CleanupWorkspace",
    "safe_shutdown": "SafeShutdown",
}


def _simulation_progress(step: str, step_progress: float | None) -> float:
    if step == "SafeShutdown":
        return 1.0
    try:
        index = _SIMULATION_STEPS.index(step)
    except ValueError:
        index = 0
    local = max(0.0, min(float(step_progress or 0.0), 1.0))
    return min((index + local) / len(_SIMULATION_STEPS), 0.99)


async def execute_workflow_run(run_id: int) -> dict:
    run = workflow_runs.get_workflow_run(run_id)
    if not run:
        record_error_event(
            layer="langgraph_workflow",
            component="workflow_execution_service",
            operation="execute_workflow_run",
            severity="error",
            error_type="WorkflowRunNotFound",
            error_message="Workflow run was not found.",
            metadata={"run_id": run_id},
        )
        return {"status": "error", "reason": "run_not_found", "run_id": run_id}
    pending = database.get_pending_action(run.get("pending_id"))
    pending_payload = (pending or {}).get("payload") or {}
    pending_data = pending_payload.get("data") or {}
    session_id = pending_data.get("session_id")
    agent_run_id = pending_data.get("agent_run_id")
    if run.get("status") != "queued":
        record_error_event(
            session_id=session_id,
            agent_run_id=agent_run_id,
            layer="langgraph_workflow",
            component="workflow_execution_service",
            operation="execute_workflow_run",
            severity="warning",
            error_type="WorkflowRunNotQueued",
            error_message="Workflow run is not queued.",
            metadata={"run_id": run_id, "run_status": run.get("status")},
        )
        return {
            "status": "error",
            "reason": "run_not_queued",
            "run_id": run_id,
            "run_status": run.get("status"),
        }
    if not workflow_runs.claim_workflow_run(run_id):
        record_error_event(
            session_id=session_id,
            agent_run_id=agent_run_id,
            layer="langgraph_workflow",
            component="workflow_execution_service",
            operation="claim_workflow_run",
            severity="warning",
            error_type="WorkflowRunClaimFailed",
            error_message="Workflow run could not be claimed.",
            metadata={"run_id": run_id},
        )
        return {"status": "error", "reason": "run_claim_failed", "run_id": run_id}

    strain_id = run.get("strain_id")
    current = database.get_algae_status(strain_id)
    expected_generation = run.get("expected_generation")
    if not current or (
        expected_generation is not None
        and int(current.get("generation_number") or 0) != int(expected_generation)
    ):
        workflow_runs.finish_workflow_run(
            run_id,
            status="stale",
            tool_result=None,
            physical_execution=False,
            persisted=False,
            error="strain_not_found" if not current else "stale_generation",
        )
        record_error_event(
            session_id=session_id,
            agent_run_id=agent_run_id,
            layer="langgraph_workflow",
            component="workflow_execution_service",
            operation="validate_workflow_run",
            severity="error",
            error_type="StrainNotFound" if not current else "StaleGeneration",
            error_message="Workflow run validation failed before execution.",
            metadata={
                "run_id": run_id,
                "strain_id": strain_id,
                "expected_generation": expected_generation,
                "current_generation": current.get("generation_number") if current else None,
            },
        )
        return {
            "status": "error",
            "reason": "strain_not_found" if not current else "stale_generation",
            "run_id": run_id,
        }

    compiled_protocol = None
    frozen_protocol = load_frozen_protocol(pending_payload)
    if frozen_protocol is not None:
        protocol_validation = verify_frozen_protocol_payload(pending_payload)
        if not protocol_validation.get("valid"):
            workflow_runs.finish_workflow_run(
                run_id,
                status="failed",
                tool_result=None,
                physical_execution=False,
                persisted=False,
                error="protocol_hash_mismatch",
            )
            append_decision_event({
                "event": "protocol_hash_mismatch",
                "pending_id": run.get("pending_id"),
                "workflow_run_id": run_id,
                "strain_id": strain_id,
                "protocol_id": protocol_validation.get("protocol_id"),
                "protocol_hash": protocol_validation.get("protocol_hash"),
            })
            record_error_event(
                session_id=session_id,
                agent_run_id=agent_run_id,
                layer="langgraph_workflow",
                component="workflow_execution_service",
                operation="verify_frozen_protocol_payload",
                severity="error",
                error_type="ProtocolHashMismatch",
                error_message="Frozen protocol payload failed hash verification before execution.",
                metadata={"run_id": run_id, "pending_id": run.get("pending_id"), **protocol_validation},
            )
            return {"status": "error", "reason": "protocol_hash_mismatch", "run_id": run_id}
        try:
            compiled_protocol = compile_protocol_to_workflow(
                frozen_protocol,
                strain_snapshot=current,
            )
            append_decision_event({
                "event": "protocol_restored",
                "agent_run_id": agent_run_id,
                "session_id": session_id,
                "pending_id": run.get("pending_id"),
                "workflow_run_id": run_id,
                "strain_id": strain_id,
                "protocol_id": frozen_protocol.protocol_id,
                "protocol_hash": frozen_protocol.protocol_hash,
            })
        except ValueError as exc:
            reason = str(exc)
            finish_status = "stale" if reason == "TARGET_STATE_STALE" else "failed"
            workflow_runs.finish_workflow_run(
                run_id,
                status=finish_status,
                tool_result=None,
                physical_execution=False,
                persisted=False,
                error=reason,
            )
            event_name = (
                "protocol_stale_state_blocked"
                if reason == "TARGET_STATE_STALE"
                else "protocol_execution_failed"
            )
            append_decision_event({
                "event": event_name,
                "pending_id": run.get("pending_id"),
                "workflow_run_id": run_id,
                "strain_id": strain_id,
                "protocol_id": frozen_protocol.protocol_id,
                "protocol_hash": frozen_protocol.protocol_hash,
                "failure_code": reason,
            })
            return {"status": "error", "reason": reason, "run_id": run_id}

    append_decision_event({
        "event": "workflow_run_started",
        "pending_id": run.get("pending_id"),
        "workflow_run_id": run_id,
        "strain_id": strain_id,
        "execution_mode": run.get("execution_mode"),
    })
    append_decision_event({
        "event": "workflow_started",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "pending_id": run.get("pending_id"),
        "workflow_run_id": run_id,
        "strain_id": strain_id,
        "execution_mode": run.get("execution_mode"),
        "protocol_id": frozen_protocol.protocol_id if frozen_protocol else None,
        "protocol_hash": frozen_protocol.protocol_hash if frozen_protocol else None,
    })
    record_run_event(
        agent_run_id,
        session_id=session_id,
        event_type="workflow_execution_started",
        layer="langgraph_workflow",
        payload={
            "pending_id": run.get("pending_id"),
            "workflow_run_id": run_id,
            "strain_id": strain_id,
            "execution_mode": run.get("execution_mode"),
        },
    )
    if frozen_protocol is not None:
        append_decision_event({
            "event": "protocol_execution_started",
            "pending_id": run.get("pending_id"),
            "workflow_run_id": run_id,
            "strain_id": strain_id,
            "protocol_id": frozen_protocol.protocol_id,
            "protocol_hash": frozen_protocol.protocol_hash,
        })
    canonical_run_id = f"workflow:{run_id}"
    initial_event = control_plane_db.append_event(
        canonical_run_id,
        "simulation_started",
        phase="CheckSchedule",
        payload={
            "step": "CheckSchedule",
            "status": "running",
            "progress": 0.0,
            "message": "传代仿真已启动",
            "simulation_only": True,
        },
    )
    initial_step_event = control_plane_db.append_event(
        canonical_run_id,
        "simulation_step_started",
        phase="CheckSchedule",
        payload={
            "step": "CheckSchedule",
            "status": "running",
            "step_progress": 0.0,
            "progress": 0.0,
            "message": "正在检查传代周期",
            "snapshot": {},
        },
    )
    workflow_runs.update_workflow_simulation_state(
        run_id,
        current_step="CheckSchedule",
        progress=0.0,
        hardware_state=None,
        event_sequence=int(initial_step_event["sequence"]),
        event_at=str(initial_step_event["created_at"]),
    )
    sink_state: dict[str, Any] = {"last_step": "CheckSchedule", "completed_steps": set()}
    sink_lock = threading.Lock()

    def persist_simulation_event(hardware_event: dict[str, Any]) -> None:
        with sink_lock:
            action = str(hardware_event.get("action") or "")
            step_key = str(hardware_event.get("task_id") or action)
            step = _ACTION_TO_STEP.get(
                step_key,
                str(hardware_event.get("step") or action or "Simulation"),
            )
            status = str(hardware_event.get("status") or "running")
            step_progress = hardware_event.get("progress")
            overall_progress = _simulation_progress(step, step_progress)
            payload = {
                "step": step,
                "device": hardware_event.get("device"),
                "action": action,
                "task_id": hardware_event.get("task_id"),
                "motion": hardware_event.get("motion"),
                "motion_command": hardware_event.get("motion_command"),
                "status": status,
                "step_progress": step_progress,
                "progress": overall_progress,
                "message": hardware_event.get("message"),
                "snapshot": hardware_event.get("snapshot") or {},
                "device_sequence": hardware_event.get("sequence"),
                "device_timestamp": hardware_event.get("timestamp"),
            }
            last_step = sink_state.get("last_step")
            completed_steps = sink_state["completed_steps"]
            if last_step != step:
                if last_step and last_step not in completed_steps:
                    control_plane_db.append_event(
                        canonical_run_id,
                        "simulation_step_completed",
                        phase=last_step,
                        payload={
                            **payload,
                            "step": last_step,
                            "status": "completed",
                            "step_progress": 1.0,
                            "progress": _simulation_progress(last_step, 1.0),
                            "message": f"{last_step} completed",
                        },
                    )
                    completed_steps.add(last_step)
                if step == "CleanupWorkspace" and "RecordExperiment" not in completed_steps:
                    record_payload = {
                        **payload,
                        "step": "RecordExperiment",
                        "status": "running",
                        "step_progress": 0.0,
                        "progress": _simulation_progress("RecordExperiment", 0.0),
                        "message": "正在写入仿真实验记录",
                    }
                    control_plane_db.append_event(
                        canonical_run_id,
                        "simulation_step_started",
                        phase="RecordExperiment",
                        payload=record_payload,
                    )
                    control_plane_db.append_event(
                        canonical_run_id,
                        "simulation_step_completed",
                        phase="RecordExperiment",
                        payload={
                            **record_payload,
                            "status": "completed",
                            "step_progress": 1.0,
                            "progress": _simulation_progress("RecordExperiment", 1.0),
                            "message": "仿真实验记录已生成",
                        },
                    )
                    completed_steps.add("RecordExperiment")
                control_plane_db.append_event(
                    canonical_run_id,
                    "simulation_step_started",
                    phase=step,
                    payload={**payload, "status": "running"},
                )
                sink_state["last_step"] = step
            event_type = (
                "simulation_fault"
                if status == "failed"
                else "simulation_step_completed"
                if status == "completed"
                else "simulation_snapshot"
            )
            persisted_event = control_plane_db.append_event(
                canonical_run_id,
                event_type,
                phase=step,
                level="error" if status == "failed" else "info",
                payload=payload,
            )
            if status == "completed":
                completed_steps.add(step)
            workflow_runs.update_workflow_simulation_state(
                run_id,
                current_step=step,
                progress=overall_progress,
                hardware_state=payload["snapshot"],
                event_sequence=int(persisted_event["sequence"]),
                event_at=str(persisted_event["created_at"]),
            )

    result = await _run_subculture_workflow(
        strain_id,
        execution_mode=run.get("execution_mode"),
        compiled_protocol=compiled_protocol,
        event_sink=persist_simulation_event,
    )
    if result.get("status") != "success":
        final_snapshot = result.get("hardware_state") or {}
        final_step = "SafeShutdown" if (result.get("last_error") or {}) else str(sink_state.get("last_step") or "Simulation")
        final_event = control_plane_db.append_event(
            canonical_run_id,
            "simulation_safe_shutdown" if final_step == "SafeShutdown" else "simulation_fault",
            phase=final_step,
            level="error",
            payload={
                "step": final_step,
                "status": "failed",
                "progress": 1.0,
                "message": result.get("message") or "传代仿真失败",
                "snapshot": final_snapshot,
                "error": result.get("last_error"),
            },
        )
        workflow_runs.update_workflow_simulation_state(
            run_id,
            current_step=final_step,
            progress=1.0,
            hardware_state=final_snapshot,
            event_sequence=int(final_event["sequence"]),
            event_at=str(final_event["created_at"]),
        )
        workflow_runs.finish_workflow_run(
            run_id,
            status="failed",
            tool_result=result,
            physical_execution=bool(result.get("physical_execution")),
            persisted=False,
            error=result.get("message") or "workflow_failed",
        )
        append_decision_event({
            "event": "workflow_run_failed",
            "pending_id": run.get("pending_id"),
            "workflow_run_id": run_id,
            "strain_id": strain_id,
            "reason": result.get("message") or "workflow_failed",
        })
        record_error_event(
            session_id=session_id,
            agent_run_id=agent_run_id,
            layer="langgraph_workflow",
            component="workflow_execution_service",
            operation="run_force_subculture_workflow",
            severity="error",
            error_type="WorkflowFailed",
            error_message=result.get("message") or "workflow_failed",
            metadata={"run_id": run_id, "strain_id": strain_id, "tool_result": result},
        )
        record_run_event(
            agent_run_id,
            session_id=session_id,
            event_type="workflow_finished",
            layer="langgraph_workflow",
            payload={"workflow_run_id": run_id, "status": "failed"},
        )
        if frozen_protocol is not None:
            append_decision_event({
                "event": "protocol_execution_failed",
                "pending_id": run.get("pending_id"),
                "workflow_run_id": run_id,
                "strain_id": strain_id,
                "protocol_id": frozen_protocol.protocol_id,
                "protocol_hash": frozen_protocol.protocol_hash,
                "failure_code": result.get("message") or "workflow_failed",
            })
        return {
            "status": "error",
            "reason": result.get("message") or "workflow_failed",
            "run_id": run_id,
            "tool_result": result,
        }

    persisted, persist_reason = commit_workflow_result(
        strain_id=strain_id,
        expected_generation=expected_generation,
        tool_result=result,
    )
    result = {**result, "persisted": persisted}
    completed_event = control_plane_db.append_event(
        canonical_run_id,
        "simulation_completed",
        phase="CleanupWorkspace",
        payload={
            "step": "CleanupWorkspace",
            "status": "completed",
            "progress": 1.0,
            "message": "传代仿真完成，等待人工完成确认",
            "snapshot": result.get("hardware_state") or {},
            "simulation_only": True,
        },
    )
    workflow_runs.update_workflow_simulation_state(
        run_id,
        current_step="CleanupWorkspace",
        progress=1.0,
        hardware_state=result.get("hardware_state") or {},
        event_sequence=int(completed_event["sequence"]),
        event_at=str(completed_event["created_at"]),
    )
    workflow_runs.finish_workflow_run(
        run_id,
        status="succeeded",
        tool_result=result,
        physical_execution=bool(result.get("physical_execution")),
        persisted=persisted,
        error=None if persisted or persist_reason == "simulation_not_persisted" else persist_reason,
    )
    database.mark_pending_executed(run.get("pending_id"), {
        "status": "success",
        "workflow_run_id": run_id,
        "physical_execution": bool(result.get("physical_execution")),
        "persisted": persisted,
        "tool_result": result,
    })
    append_decision_event({
        "event": "workflow_run_succeeded",
        "pending_id": run.get("pending_id"),
        "workflow_run_id": run_id,
        "strain_id": strain_id,
        "physical_execution": bool(result.get("physical_execution")),
        "persisted": persisted,
    })
    append_decision_event({
        "event": "workflow_succeeded",
        "agent_run_id": agent_run_id,
        "session_id": session_id,
        "pending_id": run.get("pending_id"),
        "workflow_run_id": run_id,
        "strain_id": strain_id,
        "physical_execution": bool(result.get("physical_execution")),
        "persisted": persisted,
        "protocol_id": frozen_protocol.protocol_id if frozen_protocol else None,
        "protocol_hash": frozen_protocol.protocol_hash if frozen_protocol else None,
    })
    record_run_event(
        agent_run_id,
        session_id=session_id,
        event_type="workflow_finished",
        layer="langgraph_workflow",
        payload={
            "workflow_run_id": run_id,
            "status": "succeeded",
            "physical_execution": bool(result.get("physical_execution")),
            "persisted": persisted,
        },
    )
    if persisted:
        append_decision_event({
            "event": "workflow_result_persisted",
            "pending_id": run.get("pending_id"),
            "workflow_run_id": run_id,
            "strain_id": strain_id,
        })
    if frozen_protocol is not None:
        append_decision_event({
            "event": "protocol_execution_completed",
            "pending_id": run.get("pending_id"),
            "workflow_run_id": run_id,
            "strain_id": strain_id,
            "protocol_id": frozen_protocol.protocol_id,
            "protocol_hash": frozen_protocol.protocol_hash,
            "persisted": persisted,
        })
    return {
        "status": "success",
        "run_id": run_id,
        "run_status": "succeeded",
        "tool_result": result,
        "physical_execution": bool(result.get("physical_execution")),
        "persisted": persisted,
    }
