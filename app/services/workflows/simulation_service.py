from __future__ import annotations

import copy
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.graph import build_subculture_workflow
from app.hardware import SimulatedHardware


SUPPORTED_FAULTS = {
    "uv_failure",
    "media_empty",
    "pump_a_blocked",
    "pump_b_blocked",
    "valve_failure",
    "incubator_failure",
    "spectrophotometer_timeout",
    "blank_calibration_failure",
    "invalid_absorbance",
    "liquid_handler_unavailable",
    "insufficient_medium",
    "aspirate_dispense_failure",
    "incubator_setpoint_failure",
}

STEP_ORDER = [
    "CheckSchedule",
    "ManualLoadSpectrophotometer",
    "BlankSpectrophotometer",
    "MeasureAbsorbance",
    "ManualLoadLiquidHandler",
    "LoadMaterials",
    "DispenseMedium",
    "TransferSeedCulture",
    "MixAndSeal",
    "ManualMoveToIncubator",
    "MoveToIncubator",
    "RecordExperiment",
    "CleanupWorkspace",
]

_STEP_BY_ACTION = {
    "load_spectrophotometer": "ManualLoadSpectrophotometer",
    "blank": "BlankSpectrophotometer",
    "measure_absorbance": "MeasureAbsorbance",
    "load_liquid_handler": "ManualLoadLiquidHandler",
    "check": "LoadMaterials",
    "dispense_medium": "DispenseMedium",
    "transfer_seed": "TransferSeedCulture",
    "mix_and_seal": "MixAndSeal",
    "move_to_incubator": "ManualMoveToIncubator",
    "configure": "MoveToIncubator",
    "cleanup": "CleanupWorkspace",
    "safe_shutdown": "SafeShutdown",
}


class SimulationRunNotFound(KeyError):
    pass


class ManualTaskConflict(RuntimeError):
    pass


@dataclass
class _RunContext:
    hardware: SimulatedHardware
    graph: Any
    config: dict[str, Any]
    initial_state: dict[str, Any]
    decisions: dict[str, str] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    return copy.deepcopy(value) if isinstance(value, dict) else None


class SimulationRunStore:
    """In-process SIL runner with one FIFO-scheduled virtual workcell."""

    def __init__(self, max_runs: int = 100, manual_delay_seconds: float = 0.8) -> None:
        self.max_runs = max_runs
        self.manual_delay_seconds = max(float(manual_delay_seconds), 0.0)
        self._lock = threading.RLock()
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._contexts: dict[str, _RunContext] = {}
        self._queue: deque[str] = deque()
        self._active_run_id: str | None = None

    def create(
        self,
        *,
        strain_id: str,
        generation_number: int,
        fault: str | None = None,
        fault_scenario: str | None = None,
        interaction_mode: str = "legacy",
        wavelength_nm: float = 680.0,
        simulated_absorbance: float = 0.8,
        delay_seconds: float = 0.12,
    ) -> dict[str, Any]:
        selected_fault = fault_scenario or fault
        if selected_fault and selected_fault not in SUPPORTED_FAULTS:
            raise ValueError(f"Unsupported simulation fault: {selected_fault}")
        if interaction_mode not in {"verification", "demo", "legacy"}:
            raise ValueError("interaction_mode must be verification or demo")
        if not 190.0 <= float(wavelength_nm) <= 1100.0:
            raise ValueError("wavelength_nm must be between 190 and 1100")
        if not 0.0 <= float(simulated_absorbance) <= 4.0:
            raise ValueError("simulated_absorbance must be between 0 and 4")

        run_id = str(uuid.uuid4())
        created_at = _now()
        hardware = SimulatedHardware(
            delay_seconds=delay_seconds,
            faults=[selected_fault] if selected_fault else [],
            event_sink=lambda event: self._append_event(run_id, event),
        )
        initial_state = {
            "messages": [],
            "run_id": run_id,
            "execution_mode": "simulation",
            "interaction_mode": interaction_mode,
            "strain_id": strain_id,
            "generation_number": int(generation_number),
            "inoculation_timestamp": "",
            "source_reactor_id": "Reactor_A",
            "target_reactor_id": "Reactor_B",
            "media_target_volume": 150.0,
            "seed_target_volume": 1.0,
            "wavelength_nm": float(wavelength_nm),
            "simulated_absorbance": float(simulated_absorbance),
            "measured_absorbance": None,
            "days_since_last_subculture": 6,
            "current_step": "START",
            "hardware_logs": [],
            "simulation_events": [],
            "hardware_state": hardware.snapshot(),
            "workflow_status": "IDLE",
            "pending_manual_task": None,
            "last_error": None,
        }
        record = {
            "run_id": run_id,
            "execution_mode": "simulation",
            "interaction_mode": interaction_mode,
            "status": "QUEUED",
            "workflow_status": "IDLE",
            "current_step": "START",
            "strain_id": strain_id,
            "generation_number": int(generation_number),
            "fault_scenario": selected_fault,
            "wavelength_nm": float(wavelength_nm),
            "simulated_absorbance": float(simulated_absorbance),
            "measured_absorbance": None,
            "created_at": created_at,
            "updated_at": created_at,
            "hardware_state": hardware.snapshot(),
            "events": [],
            "hardware_logs": [],
            "pending_manual_task": None,
            "queue_position": None,
            "overall_progress": 0.0,
            "last_error": None,
            "checkpointed": True,
        }
        graph = build_subculture_workflow(hardware, checkpointer=InMemorySaver())
        context = _RunContext(
            hardware=hardware,
            graph=graph,
            config={"configurable": {"thread_id": run_id}},
            initial_state=initial_state,
        )
        with self._lock:
            self._runs[run_id] = record
            self._contexts[run_id] = context
            self._queue.append(run_id)
            self._trim_locked()
            self._update_queue_positions_locked()
            self._launch_next_locked()
            return copy.deepcopy(self._runs[run_id])

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._runs.get(run_id)
            return copy.deepcopy(record) if record else None

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 50), self.max_runs))
        with self._lock:
            records = list(self._runs.values())[-safe_limit:]
            return [copy.deepcopy(item) for item in reversed(records)]

    def resolve_manual_task(
        self,
        run_id: str,
        task_id: str,
        *,
        resolution: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        if resolution not in {"completed", "failed"}:
            raise ValueError("resolution must be completed or failed")
        with self._lock:
            record = self._runs.get(run_id)
            context = self._contexts.get(run_id)
            if record is None or context is None:
                raise SimulationRunNotFound(run_id)
            previous = context.decisions.get(task_id)
            if previous:
                if previous != resolution:
                    raise ManualTaskConflict("Manual task already resolved differently")
                response = copy.deepcopy(record)
                response["idempotent"] = True
                return response
            pending = record.get("pending_manual_task") or {}
            if record.get("status") != "WAITING_MANUAL" or pending.get("task_id") != task_id:
                raise ManualTaskConflict("Manual task is not currently pending")
            context.decisions[task_id] = resolution
            record["status"] = "RUNNING"
            record["workflow_status"] = "RUNNING"
            record["pending_manual_task"] = None
            record["updated_at"] = _now()
            thread = threading.Thread(
                target=self._invoke,
                args=(run_id, Command(resume={"resolution": resolution, "note": note or ""})),
                name=f"subculture-resume-{run_id[:8]}",
                daemon=True,
            )
            thread.start()
            response = copy.deepcopy(record)
            response["idempotent"] = False
            return response

    def _trim_locked(self) -> None:
        while len(self._runs) > self.max_runs:
            removable = next(
                (
                    run_id
                    for run_id, record in self._runs.items()
                    if run_id != self._active_run_id
                    and run_id not in self._queue
                    and record["status"] in {"COMPLETED", "FAILED", "SKIPPED"}
                ),
                None,
            )
            if removable is None:
                break
            self._runs.pop(removable, None)
            self._contexts.pop(removable, None)

    def _update_queue_positions_locked(self) -> None:
        positions = {run_id: index + 1 for index, run_id in enumerate(self._queue)}
        for run_id, record in self._runs.items():
            record["queue_position"] = positions.get(run_id)
        if self._active_run_id and self._active_run_id in self._runs:
            self._runs[self._active_run_id]["queue_position"] = 0

    def _launch_next_locked(self) -> None:
        if self._active_run_id is not None or not self._queue:
            return
        run_id = self._queue.popleft()
        self._active_run_id = run_id
        record = self._runs[run_id]
        record.update(
            {
                "status": "RUNNING",
                "workflow_status": "RUNNING",
                "updated_at": _now(),
            }
        )
        self._update_queue_positions_locked()
        context = self._contexts[run_id]
        thread = threading.Thread(
            target=self._invoke,
            args=(run_id, context.initial_state),
            name=f"subculture-simulation-{run_id[:8]}",
            daemon=True,
        )
        thread.start()

    def _append_event(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if not record:
                return
            record["events"].append(event)
            record["hardware_state"] = event.get("snapshot") or record["hardware_state"]
            record["current_step"] = _STEP_BY_ACTION.get(
                str(event.get("action")), record["current_step"]
            )
            record["overall_progress"] = self._progress(record["current_step"])
            record["updated_at"] = _now()

    @staticmethod
    def _progress(step: str) -> float:
        if step == "SafeShutdown":
            return 1.0
        try:
            return (STEP_ORDER.index(step) + 1) / len(STEP_ORDER)
        except ValueError:
            return 0.0

    def _invoke(self, run_id: str, graph_input: Any) -> None:
        context = self._contexts[run_id]
        try:
            result = context.graph.invoke(graph_input, config=context.config)
            pending = _interrupt_payload(result)
            if pending:
                with self._lock:
                    record = self._runs[run_id]
                    record.update(
                        {
                            "status": "WAITING_MANUAL",
                            "workflow_status": "WAITING_MANUAL",
                            "current_step": pending.get("step", record["current_step"]),
                            "pending_manual_task": pending,
                            "overall_progress": self._progress(
                                pending.get("step", record["current_step"])
                            ),
                            "hardware_state": context.hardware.snapshot(),
                            "updated_at": _now(),
                        }
                    )
                    mode = record["interaction_mode"]
                if mode == "demo":
                    threading.Thread(
                        target=self._auto_resolve,
                        args=(run_id, str(pending["task_id"])),
                        name=f"subculture-auto-{run_id[:8]}",
                        daemon=True,
                    ).start()
                return

            workflow_status = result.get("workflow_status", "FAILED")
            terminal_status = (
                "COMPLETED"
                if workflow_status == "SUCCESS"
                else "SKIPPED"
                if workflow_status == "SKIPPED"
                else "FAILED"
            )
            with self._lock:
                record = self._runs[run_id]
                record.update(
                    {
                        "status": terminal_status,
                        "workflow_status": workflow_status,
                        "current_step": result.get("current_step", record["current_step"]),
                        "generation_number": result.get(
                            "generation_number", record["generation_number"]
                        ),
                        "measured_absorbance": result.get("measured_absorbance"),
                        "hardware_state": result.get(
                            "hardware_state", context.hardware.snapshot()
                        ),
                        "hardware_logs": result.get("hardware_logs", []),
                        "pending_manual_task": None,
                        "overall_progress": 1.0,
                        "last_error": result.get("last_error"),
                        "updated_at": _now(),
                    }
                )
                self._finish_active_locked(run_id)
        except Exception as exc:
            context.hardware.safe_shutdown(str(exc))
            with self._lock:
                record = self._runs[run_id]
                record.update(
                    {
                        "status": "FAILED",
                        "workflow_status": "FAILED",
                        "current_step": "SafeShutdown",
                        "hardware_state": context.hardware.snapshot(),
                        "pending_manual_task": None,
                        "overall_progress": 1.0,
                        "last_error": {
                            "code": "SIMULATION_RUNTIME_ERROR",
                            "message": str(exc),
                        },
                        "updated_at": _now(),
                    }
                )
                self._finish_active_locked(run_id)

    def _auto_resolve(self, run_id: str, task_id: str) -> None:
        if self.manual_delay_seconds:
            time.sleep(self.manual_delay_seconds)
        try:
            self.resolve_manual_task(
                run_id,
                task_id,
                resolution="completed",
                note="Demo mode automatic confirmation",
            )
        except (SimulationRunNotFound, ManualTaskConflict):
            return

    def _finish_active_locked(self, run_id: str) -> None:
        if self._active_run_id == run_id:
            self._active_run_id = None
        self._update_queue_positions_locked()
        self._launch_next_locked()


simulation_runs = SimulationRunStore()
