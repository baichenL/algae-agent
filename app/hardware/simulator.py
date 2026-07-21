from __future__ import annotations

import copy
import threading
import time
from datetime import UTC, datetime
from typing import Any, Callable, Iterable

from app.hardware.base import HardwareResult
from app.hardware.devices import (
    SimulatedIncubator,
    SimulatedLiquidHandler,
    SimulatedSpectrophotometer,
)


EventSink = Callable[[dict[str, Any]], None]


class SimulatedHardware:
    """Stateful digital twin for the automated subculture workstation.

    The simulator enforces simple mass-balance and device-interlock rules. It
    emits snapshots during long operations so a UI can animate the same state
    that drives workflow decisions.
    """

    def __init__(
        self,
        *,
        delay_seconds: float = 0.12,
        faults: Iterable[str] | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.delay_seconds = max(float(delay_seconds), 0.0)
        self.faults = set(faults or [])
        self.event_sink = event_sink
        self._lock = threading.RLock()
        self._sequence = 0
        self._events: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {
            "phase": "IDLE",
            "uv": {"status": "OFF", "progress": 0.0},
            "media_reservoir": {"volume_ml": 1000.0, "capacity_ml": 1000.0},
            "source_reactor": {
                "id": "Reactor_A",
                "volume_ml": 200.0,
                "capacity_ml": 250.0,
                "sealed": True,
            },
            "target_reactor": {
                "id": "Reactor_B",
                "volume_ml": 0.0,
                "capacity_ml": 250.0,
                "sealed": False,
            },
            "media_pump": {"status": "OFF", "progress": 0.0},
            "seed_pump": {"status": "OFF", "progress": 0.0},
            "valve": {"status": "OPEN"},
            "incubator": {
                "status": "STANDBY",
                "temperature_c": 22.0,
                "light_lux": 0.0,
            },
            "spectrophotometer": {
                "status": "STANDBY",
                "wavelength_nm": 680.0,
                "absorbance": None,
                "blanked": False,
            },
            "liquid_handler": {
                "status": "STANDBY",
                "progress": 0.0,
                "deck_loaded": False,
            },
            "sample": {
                "container": "source_flask",
                "container_type": "erlenmeyer_flask",
                "location": "manual_zone",
            },
            "alarm": None,
        }
        self.spectrophotometer = SimulatedSpectrophotometer(self)
        self.liquid_handler = SimulatedLiquidHandler(self)
        self.incubator_controller = SimulatedIncubator(self)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def _pause(self) -> None:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)

    def _emit(
        self,
        *,
        device: str,
        action: str,
        message: str,
        status: str = "running",
        progress: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "timestamp": datetime.now(UTC).isoformat(),
                "device": device,
                "action": action,
                "message": message,
                "status": status,
                "progress": progress,
                "snapshot": copy.deepcopy(self._state),
            }
            self._events.append(event)
        if self.event_sink:
            self.event_sink(copy.deepcopy(event))
        return event

    def _result(
        self,
        start_index: int,
        *,
        ok: bool,
        device: str,
        action: str,
        message: str,
        error_code: str | None = None,
    ) -> HardwareResult:
        with self._lock:
            events = copy.deepcopy(self._events[start_index:])
        return HardwareResult(
            ok=ok,
            device=device,
            action=action,
            message=message,
            state=self.snapshot(),
            events=events,
            error_code=error_code,
        )

    def _fail(
        self,
        start_index: int,
        *,
        device: str,
        action: str,
        message: str,
        error_code: str,
    ) -> HardwareResult:
        with self._lock:
            self._state["phase"] = "FAULT"
            self._state["alarm"] = {"code": error_code, "message": message}
        self._emit(
            device=device,
            action=action,
            message=message,
            status="failed",
        )
        return self._result(
            start_index,
            ok=False,
            device=device,
            action=action,
            message=message,
            error_code=error_code,
        )

    def relocate_sample(
        self,
        *,
        task_id: str,
        location: str,
        container: str,
    ) -> HardwareResult:
        start = len(self._events)
        with self._lock:
            self._state["phase"] = "MANUAL_TRANSFER"
            self._state["sample"].update(
                {"location": "in_transit", "container": container}
            )
        self._emit(
            device="human_operator",
            action=task_id,
            message=f"Moving {container} to {location}",
            progress=0.5,
        )
        self._pause()
        with self._lock:
            self._state["sample"]["location"] = location
            if location == "liquid_handler":
                self._state["liquid_handler"]["deck_loaded"] = True
        self._emit(
            device="human_operator",
            action=task_id,
            message=f"{container} placed at {location}",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="human_operator",
            action=task_id,
            message="Manual transfer completed",
        )

    def blank_spectrophotometer(self, wavelength_nm: float) -> HardwareResult:
        start = len(self._events)
        if "spectrophotometer_timeout" in self.faults:
            return self._fail(
                start,
                device="spectrophotometer",
                action="blank",
                message="Spectrophotometer did not respond during blank calibration",
                error_code="SPECTROPHOTOMETER_TIMEOUT",
            )
        if "blank_calibration_failure" in self.faults:
            return self._fail(
                start,
                device="spectrophotometer",
                action="blank",
                message="Blank calibration was rejected",
                error_code="BLANK_CALIBRATION_FAILED",
            )
        with self._lock:
            self._state["phase"] = "BLANKING"
            self._state["spectrophotometer"].update(
                {
                    "status": "RUNNING",
                    "wavelength_nm": float(wavelength_nm),
                    "blanked": False,
                }
            )
        self._emit(
            device="spectrophotometer",
            action="blank",
            message=f"Blanking at {wavelength_nm:.0f} nm",
            progress=0.5,
        )
        self._pause()
        with self._lock:
            self._state["spectrophotometer"].update(
                {"status": "READY", "blanked": True}
            )
        self._emit(
            device="spectrophotometer",
            action="blank",
            message="Blank calibration completed",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="spectrophotometer",
            action="blank",
            message="Blank calibration completed",
        )

    def measure_absorbance(
        self,
        *,
        wavelength_nm: float,
        expected_absorbance: float,
    ) -> HardwareResult:
        start = len(self._events)
        with self._lock:
            blanked = bool(self._state["spectrophotometer"]["blanked"])
        if not blanked:
            return self._fail(
                start,
                device="spectrophotometer",
                action="measure_absorbance",
                message="Absorbance measurement requires a valid blank",
                error_code="SPECTROPHOTOMETER_NOT_BLANKED",
            )
        if "invalid_absorbance" in self.faults:
            return self._fail(
                start,
                device="spectrophotometer",
                action="measure_absorbance",
                message="Spectrophotometer returned an invalid absorbance value",
                error_code="INVALID_ABSORBANCE",
            )
        with self._lock:
            self._state["phase"] = "MEASURING_OD"
            self._state["spectrophotometer"]["status"] = "RUNNING"
        self._emit(
            device="spectrophotometer",
            action="measure_absorbance",
            message=f"Measuring absorbance at {wavelength_nm:.0f} nm",
            progress=0.5,
        )
        self._pause()
        with self._lock:
            self._state["spectrophotometer"].update(
                {
                    "status": "COMPLETED",
                    "wavelength_nm": float(wavelength_nm),
                    "absorbance": float(expected_absorbance),
                }
            )
        self._emit(
            device="spectrophotometer",
            action="measure_absorbance",
            message=f"Absorbance measured: {expected_absorbance:.3f}",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="spectrophotometer",
            action="measure_absorbance",
            message=f"OD result recorded: {expected_absorbance:.3f}",
        )

    def sterilize_workspace(self, duration_min: float) -> HardwareResult:
        start = len(self._events)
        if "uv_failure" in self.faults:
            return self._fail(
                start,
                device="uv_sterilizer",
                action="sterilize",
                message="UV sterilizer failed to start",
                error_code="UV_START_FAILURE",
            )
        with self._lock:
            self._state["phase"] = "STERILIZING"
            self._state["uv"]["status"] = "ON"
        for index in range(1, 6):
            progress = index / 5
            with self._lock:
                self._state["uv"]["progress"] = progress
            self._emit(
                device="uv_sterilizer",
                action="sterilize",
                message=f"UV sterilization {progress:.0%}",
                progress=progress,
            )
            self._pause()
        with self._lock:
            self._state["uv"] = {"status": "OFF", "progress": 1.0}
        self._emit(
            device="uv_sterilizer",
            action="sterilize",
            message=f"Workspace sterilized (virtual {duration_min:g} min cycle)",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="uv_sterilizer",
            action="sterilize",
            message="Workspace sterilization completed",
        )

    def check_materials(self, expected_media_ml: float) -> HardwareResult:
        start = len(self._events)
        if "liquid_handler_unavailable" in self.faults:
            return self._fail(
                start,
                device="liquid_handler",
                action="check",
                message="Liquid handler is unavailable",
                error_code="LIQUID_HANDLER_UNAVAILABLE",
            )
        with self._lock:
            self._state["phase"] = "CHECKING_MATERIALS"
            self._state["liquid_handler"]["status"] = "CHECKING"
            available = float(self._state["media_reservoir"]["volume_ml"])
        if (
            "media_empty" in self.faults
            or "insufficient_medium" in self.faults
            or available < expected_media_ml
        ):
            return self._fail(
                start,
                device="material_sensors",
                action="check",
                message=(
                    f"Insufficient medium: required {expected_media_ml:.1f} mL, "
                    f"available {available:.1f} mL"
                ),
                error_code="INSUFFICIENT_MEDIUM",
            )
        self._emit(
            device="material_sensors",
            action="check",
            message=f"Materials ready; {available:.1f} mL medium available",
            status="completed",
            progress=1.0,
        )
        self._pause()
        with self._lock:
            self._state["liquid_handler"]["status"] = "READY"
        return self._result(
            start,
            ok=True,
            device="material_sensors",
            action="check",
            message="Material checks passed",
        )

    def dispense_medium(self, target: str, volume_ml: float) -> HardwareResult:
        start = len(self._events)
        with self._lock:
            reservoir = float(self._state["media_reservoir"]["volume_ml"])
            target_volume = float(self._state["target_reactor"]["volume_ml"])
            capacity = float(self._state["target_reactor"]["capacity_ml"])
        if "pump_a_blocked" in self.faults:
            return self._fail(
                start,
                device="media_pump",
                action="dispense_medium",
                message="Media pump is blocked",
                error_code="MEDIA_PUMP_BLOCKED",
            )
        if reservoir < volume_ml or target_volume + volume_ml > capacity:
            return self._fail(
                start,
                device="media_pump",
                action="dispense_medium",
                message="Requested medium transfer violates volume limits",
                error_code="MEDIA_VOLUME_LIMIT",
            )
        with self._lock:
            self._state["phase"] = "DISPENSING_MEDIUM"
            self._state["target_reactor"]["id"] = target
            self._state["media_pump"]["status"] = "RUNNING"
            self._state["liquid_handler"]["status"] = "RUNNING"
        increments = 10
        for index in range(1, increments + 1):
            moved = volume_ml / increments
            with self._lock:
                self._state["media_reservoir"]["volume_ml"] -= moved
                self._state["target_reactor"]["volume_ml"] += moved
                self._state["media_pump"]["progress"] = index / increments
            self._emit(
                device="media_pump",
                action="dispense_medium",
                message=f"Dispensing medium to {target}",
                progress=index / increments,
            )
            self._pause()
            if "aspirate_dispense_failure" in self.faults and index == 4:
                with self._lock:
                    self._state["media_pump"]["status"] = "OFF"
                    self._state["liquid_handler"]["status"] = "FAULT"
                return self._fail(
                    start,
                    device="liquid_handler",
                    action="dispense_medium",
                    message="Liquid handler stopped after a dispense error",
                    error_code="ASPIRATE_DISPENSE_FAILURE",
                )
        with self._lock:
            self._state["media_pump"]["status"] = "OFF"
            self._state["liquid_handler"].update(
                {"status": "READY", "progress": 1.0}
            )
        self._emit(
            device="media_pump",
            action="dispense_medium",
            message=f"Dispensed {volume_ml:.1f} mL medium",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="media_pump",
            action="dispense_medium",
            message=f"Medium transfer completed: {volume_ml:.1f} mL",
        )

    def transfer_seed(self, source: str, target: str, volume_ml: float) -> HardwareResult:
        start = len(self._events)
        with self._lock:
            source_volume = float(self._state["source_reactor"]["volume_ml"])
            target_volume = float(self._state["target_reactor"]["volume_ml"])
            capacity = float(self._state["target_reactor"]["capacity_ml"])
        if "pump_b_blocked" in self.faults:
            return self._fail(
                start,
                device="seed_pump",
                action="transfer_seed",
                message="Seed transfer pump is blocked",
                error_code="SEED_PUMP_BLOCKED",
            )
        if source_volume < volume_ml or target_volume + volume_ml > capacity:
            return self._fail(
                start,
                device="seed_pump",
                action="transfer_seed",
                message="Requested seed transfer violates volume limits",
                error_code="SEED_VOLUME_LIMIT",
            )
        with self._lock:
            self._state["phase"] = "TRANSFERRING_SEED"
            self._state["source_reactor"]["id"] = source
            self._state["target_reactor"]["id"] = target
            self._state["seed_pump"]["status"] = "RUNNING"
            self._state["liquid_handler"]["status"] = "RUNNING"
        increments = 8
        for index in range(1, increments + 1):
            moved = volume_ml / increments
            with self._lock:
                self._state["source_reactor"]["volume_ml"] -= moved
                self._state["target_reactor"]["volume_ml"] += moved
                self._state["seed_pump"]["progress"] = index / increments
            self._emit(
                device="seed_pump",
                action="transfer_seed",
                message=f"Transferring inoculum from {source} to {target}",
                progress=index / increments,
            )
            self._pause()
        with self._lock:
            self._state["seed_pump"]["status"] = "OFF"
            self._state["liquid_handler"].update(
                {"status": "READY", "progress": 1.0}
            )
        self._emit(
            device="seed_pump",
            action="transfer_seed",
            message=f"Transferred {volume_ml:.1f} mL seed culture",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="seed_pump",
            action="transfer_seed",
            message=f"Seed culture transfer completed: {volume_ml:.1f} mL",
        )

    def close_reactor(self, reactor: str) -> HardwareResult:
        start = len(self._events)
        if "valve_failure" in self.faults:
            return self._fail(
                start,
                device="target_valve",
                action="close",
                message="Target reactor valve did not reach closed position",
                error_code="VALVE_CLOSE_FAILURE",
            )
        with self._lock:
            self._state["phase"] = "SEALING"
            self._state["valve"]["status"] = "CLOSED"
            self._state["target_reactor"]["sealed"] = True
        self._emit(
            device="target_valve",
            action="close",
            message=f"{reactor} sealed",
            status="completed",
            progress=1.0,
        )
        self._pause()
        return self._result(
            start,
            ok=True,
            device="target_valve",
            action="close",
            message="Target reactor sealed",
        )

    def mix_and_seal(self, reactor: str) -> HardwareResult:
        start = len(self._events)
        with self._lock:
            self._state["phase"] = "MIXING"
            self._state["liquid_handler"]["status"] = "RUNNING"
        self._emit(
            device="liquid_handler",
            action="mix_and_seal",
            message=f"Mixing culture in {reactor}",
            progress=0.5,
        )
        self._pause()
        if "valve_failure" in self.faults:
            return self._fail(
                start,
                device="liquid_handler",
                action="mix_and_seal",
                message="Target flask could not be sealed",
                error_code="FLASK_SEAL_FAILURE",
            )
        with self._lock:
            self._state["phase"] = "SEALED"
            self._state["liquid_handler"]["status"] = "READY"
            self._state["valve"]["status"] = "CLOSED"
            self._state["target_reactor"]["sealed"] = True
        self._emit(
            device="liquid_handler",
            action="mix_and_seal",
            message=f"{reactor} mixed and sealed",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="liquid_handler",
            action="mix_and_seal",
            message="Culture mixed and target flask sealed",
        )

    def configure_incubator(self, temperature_c: float, light_lux: float) -> HardwareResult:
        start = len(self._events)
        if (
            "incubator_failure" in self.faults
            or "incubator_setpoint_failure" in self.faults
        ):
            return self._fail(
                start,
                device="incubator",
                action="configure",
                message="Incubator controller rejected the setpoint",
                error_code="INCUBATOR_SETPOINT_FAILURE",
            )
        with self._lock:
            self._state["phase"] = "INCUBATING"
            self._state["incubator"] = {
                "status": "RUNNING",
                "temperature_c": float(temperature_c),
                "light_lux": float(light_lux),
            }
            self._state["sample"].update(
                {"location": "incubator", "container": "target_flask"}
            )
        self._emit(
            device="incubator",
            action="configure",
            message=f"Incubator set to {temperature_c:.1f} C / {light_lux:.0f} lux",
            status="completed",
            progress=1.0,
        )
        self._pause()
        return self._result(
            start,
            ok=True,
            device="incubator",
            action="configure",
            message="Incubator configured",
        )

    def cleanup(self) -> HardwareResult:
        start = len(self._events)
        with self._lock:
            self._state["phase"] = "COMPLETED"
            self._state["uv"]["status"] = "OFF"
            self._state["media_pump"]["status"] = "OFF"
            self._state["seed_pump"]["status"] = "OFF"
            self._state["liquid_handler"]["status"] = "STANDBY"
            self._state["spectrophotometer"]["status"] = "STANDBY"
        self._emit(
            device="workcell",
            action="cleanup",
            message="Workcell reset; simulated subculture completed",
            status="completed",
            progress=1.0,
        )
        self._pause()
        return self._result(
            start,
            ok=True,
            device="workcell",
            action="cleanup",
            message="Cleanup completed",
        )

    def safe_shutdown(self, reason: str) -> HardwareResult:
        start = len(self._events)
        with self._lock:
            self._state["phase"] = "SAFE_SHUTDOWN"
            self._state["uv"]["status"] = "OFF"
            self._state["media_pump"]["status"] = "OFF"
            self._state["seed_pump"]["status"] = "OFF"
            self._state["liquid_handler"]["status"] = "SAFE"
            self._state["spectrophotometer"]["status"] = "SAFE"
            if self._state["incubator"]["status"] != "RUNNING":
                self._state["incubator"]["status"] = "STANDBY"
        self._emit(
            device="safety_controller",
            action="safe_shutdown",
            message=f"Safe shutdown completed: {reason}",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="safety_controller",
            action="safe_shutdown",
            message="Hardware placed in safe state",
        )
