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
from app.hardware.motion import MOTION_SCHEMA_VERSION, build_motion_command


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
        self._simulation_time_ms = 0
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
            "plate_reader": {
                "status": "STANDBY",
                "door_state": "CLOSED",
                "tray_state": "HOME",
                "wavelength_nm": 680.0,
                "absorbance": None,
                "blanked": False,
            },
            "measurement_plate": {
                "id": "measurement_plate_1",
                "plate_type": "96_well",
                "location": "plate_stack",
                "sample_well": "A1",
                "blank_well": "A2",
                "sample_volume_ml": 0.0,
                "blank_volume_ml": 0.0,
                "status": "EMPTY",
            },
            "robot_arm": {
                "status": "STANDBY",
                "pose": "HOME",
                "gripper_state": "OPEN",
            },
            "mobile_robot": {
                "status": "STANDBY",
                "location": "home",
                "pose": "HOME",
                "gripper_state": "OPEN",
            },
            "transport": {
                "carrier": None,
                "labware_id": None,
                "source_location": None,
                "target_location": None,
                "motion_phase": "IDLE",
                "progress": 0.0,
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
            "motion_schema_version": MOTION_SCHEMA_VERSION,
            "simulation_time_ms": 0,
            "entities": {
                "source_flask": {"entity_type": "culture_flask", "location": "manual_zone", "parent_id": "manual_handoff.source_slot"},
                "target_flask": {"entity_type": "culture_flask", "location": "manual_zone", "parent_id": "manual_handoff.target_slot"},
                "media_bottle": {"entity_type": "media_bottle", "location": "manual_zone", "parent_id": "manual_handoff.media_slot"},
                "measurement_plate": {"entity_type": "microplate_96", "location": "plate_stack", "parent_id": "plate_stack.slot_1"},
            },
            "attachments": {
                "source_flask": "manual_handoff.source_slot",
                "target_flask": "manual_handoff.target_slot",
                "media_bottle": "manual_handoff.media_slot",
                "measurement_plate": "plate_stack.slot_1",
            },
            "resource_occupancy": {},
            "active_commands": [],
            "device_interlocks": {
                "liquid_handler_door": "CLOSED",
                "plate_reader_door": "CLOSED",
                "plate_reader_tray": "HOME",
                "incubator_door": "CLOSED",
                "media_valve": "OPEN",
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

    def _set_attachment(self, entity_id: str, parent_id: str, location: str) -> None:
        """Move a labware entity atomically between scene-graph parents."""
        with self._lock:
            self._state["attachments"][entity_id] = parent_id
            entity = self._state["entities"].setdefault(entity_id, {"entity_type": "labware"})
            entity.update({"parent_id": parent_id, "location": location})

    def _emit(
        self,
        *,
        device: str,
        action: str,
        message: str,
        status: str = "running",
        progress: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event_metadata = copy.deepcopy(metadata or {})
            motion = event_metadata.get("motion") or self._state.get("transport") or {}
            entity_id = str(
                motion.get("labware_id")
                or ("target_flask" if action in {"dispense_medium", "transfer_seed", "mix_and_seal", "configure"} else "measurement_plate" if "plate" in action else device)
            )
            attachment_transition = event_metadata.get("attachment_transition")
            command = event_metadata.get("motion_command") or build_motion_command(
                sequence=self._sequence,
                actor_id=device,
                entity_id=entity_id,
                action_type=action,
                message=message,
                simulation_start_ms=self._simulation_time_ms,
                source_location=motion.get("source_location"),
                target_location=motion.get("target_location"),
                attachment_transition=attachment_transition,
                status=status,
                progress=progress,
            )
            self._simulation_time_ms += int(command.get("nominal_duration_ms") or 0)
            self._state["simulation_time_ms"] = self._simulation_time_ms
            self._state["motion_schema_version"] = MOTION_SCHEMA_VERSION
            self._state["active_commands"] = [copy.deepcopy(command)]
            self._state["resource_occupancy"] = {
                resource: command["command_id"] for resource in command.get("required_resources", [])
            }
            event = {
                "sequence": self._sequence,
                "timestamp": datetime.now(UTC).isoformat(),
                "device": device,
                "action": action,
                "message": message,
                "status": status,
                "progress": progress,
                "motion_command": copy.deepcopy(command),
                "snapshot": copy.deepcopy(self._state),
            }
            if event_metadata:
                event.update(event_metadata)
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
            self._state["active_commands"] = []
            self._state["resource_occupancy"] = {}
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
        carrier: str = "human_operator",
    ) -> HardwareResult:
        start = len(self._events)
        carrier_state = self._state.get(carrier) if carrier in {"robot_arm", "mobile_robot"} else None
        carrier_parent = (
            "robot_arm.gripper"
            if carrier == "robot_arm"
            else "mobile_robot.tray"
            if carrier == "mobile_robot"
            else "manual_operator.handoff_tray"
        )
        destination_parent = f"{location}.slot"
        with self._lock:
            source_location = (
                self._state["measurement_plate"]["location"]
                if container == "measurement_plate"
                else self._state["sample"]["location"]
            )
            self._state["phase"] = "MANUAL_TRANSFER"
            self._state["transport"] = {
                "carrier": carrier,
                "labware_id": container,
                "source_location": source_location,
                "target_location": location,
                "motion_phase": "PICKING",
                "progress": 0.15,
            }
            if carrier_state is not None:
                carrier_state.update({"status": "RUNNING", "pose": "PICK", "gripper_state": "OPEN"})
        self._emit(
            device=carrier,
            action="pick_labware",
            message=f"Picking {container} at {source_location}",
            progress=0.15,
            metadata={
                "task_id": task_id,
                "motion": copy.deepcopy(self._state["transport"]),
                "attachment_transition": {
                    "entity_id": container,
                    "phase": "grip_verify",
                    "from_parent": self._state["attachments"].get(container),
                    "to_parent": carrier_parent,
                },
            },
        )
        self._pause()
        with self._lock:
            self._state["transport"].update({"motion_phase": "TRANSPORTING", "progress": 0.6})
            if carrier_state is not None:
                carrier_state.update({"pose": "CARRY", "gripper_state": "CLOSED"})
        self._set_attachment(container, carrier_parent, "in_transit")
        with self._lock:
            if container == "measurement_plate":
                self._state["measurement_plate"]["location"] = "in_transit"
            else:
                self._state["sample"].update({"location": "in_transit", "container": container})
        self._emit(
            device=carrier,
            action="transport_labware",
            message=f"Transporting {container} to {location}",
            progress=0.6,
            metadata={"task_id": task_id, "motion": copy.deepcopy(self._state["transport"])},
        )
        self._pause()
        with self._lock:
            self._state["transport"].update({"motion_phase": "PLACING", "progress": 0.9})
            if carrier_state is not None:
                carrier_state.update({"pose": "PLACE", "gripper_state": "CLOSED"})
        self._emit(
            device=carrier,
            action="place_labware",
            message=f"Positioning {container} at {location}",
            progress=0.9,
            metadata={
                "task_id": task_id,
                "motion": copy.deepcopy(self._state["transport"]),
                "attachment_transition": {
                    "entity_id": container,
                    "phase": "release",
                    "from_parent": carrier_parent,
                    "to_parent": destination_parent,
                },
            },
        )
        self._pause()
        self._set_attachment(container, destination_parent, location)
        with self._lock:
            if container == "measurement_plate":
                self._state["measurement_plate"]["location"] = location
            else:
                self._state["sample"]["location"] = location
            if location == "liquid_handler":
                self._state["liquid_handler"]["deck_loaded"] = True
                if task_id == "load_liquid_handler":
                    self._state["measurement_plate"]["location"] = "liquid_handler"
                    self._set_attachment("measurement_plate", "liquid_handler.plate_slot", "liquid_handler")
            self._state["transport"].update({"motion_phase": "COMPLETED", "progress": 1.0})
            if carrier_state is not None:
                carrier_state.update({"status": "READY", "pose": "HOME", "gripper_state": "OPEN"})
                if carrier == "mobile_robot":
                    carrier_state["location"] = location
        self._emit(
            device=carrier,
            action="place_labware",
            message=f"{container} placed and verified at {location}",
            status="completed",
            progress=1.0,
            metadata={"task_id": task_id, "motion": copy.deepcopy(self._state["transport"])},
        )
        self._emit(
            device=carrier,
            action=task_id,
            message=f"Transfer completed: {container} at {location}",
            status="completed",
            progress=1.0,
            metadata={"task_id": task_id, "motion": copy.deepcopy(self._state["transport"])},
        )
        return self._result(
            start,
            ok=True,
            device=carrier,
            action=task_id,
            message="Labware transfer completed",
        )

    def prepare_measurement_plate(
        self,
        *,
        sample_volume_ml: float,
        blank_volume_ml: float,
    ) -> HardwareResult:
        start = len(self._events)
        with self._lock:
            source_volume = float(self._state["source_reactor"]["volume_ml"])
            medium_volume = float(self._state["media_reservoir"]["volume_ml"])
            plate_location = self._state["measurement_plate"]["location"]
        if source_volume < sample_volume_ml or medium_volume < blank_volume_ml:
            return self._fail(
                start,
                device="liquid_handler",
                action="prepare_measurement_plate",
                message="Insufficient liquid for the measurement plate",
                error_code="MEASUREMENT_PLATE_VOLUME_LIMIT",
            )
        if plate_location != "liquid_handler":
            return self._fail(
                start,
                device="liquid_handler",
                action="prepare_measurement_plate",
                message="Measurement plate is not loaded on the liquid handler",
                error_code="MEASUREMENT_PLATE_NOT_LOADED",
            )
        with self._lock:
            self._state["phase"] = "PREPARING_MEASUREMENT_PLATE"
            self._state["liquid_handler"].update({"status": "RUNNING", "progress": 0.25})
        self._emit(
            device="liquid_handler",
            action="prepare_measurement_plate",
            message="Aspirating culture sample for well A1",
            progress=0.25,
        )
        self._pause()
        with self._lock:
            self._state["source_reactor"]["volume_ml"] -= sample_volume_ml
            self._state["measurement_plate"]["sample_volume_ml"] = sample_volume_ml
            self._state["liquid_handler"]["progress"] = 0.6
        self._emit(
            device="liquid_handler",
            action="prepare_measurement_plate",
            message="Dispensing culture sample into well A1",
            progress=0.6,
        )
        self._pause()
        with self._lock:
            self._state["media_reservoir"]["volume_ml"] -= blank_volume_ml
            self._state["measurement_plate"].update(
                {"blank_volume_ml": blank_volume_ml, "status": "PREPARED"}
            )
            self._state["liquid_handler"].update({"status": "READY", "progress": 1.0})
        self._emit(
            device="liquid_handler",
            action="prepare_measurement_plate",
            message="Measurement plate prepared: sample A1 / blank A2",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="liquid_handler",
            action="prepare_measurement_plate",
            message="Measurement plate prepared",
        )

    def blank_spectrophotometer(self, wavelength_nm: float) -> HardwareResult:
        start = len(self._events)
        if "spectrophotometer_timeout" in self.faults:
            return self._fail(
                start,
                device="plate_reader",
                action="blank_plate",
                message="Plate reader did not respond during blank calibration",
                error_code="SPECTROPHOTOMETER_TIMEOUT",
            )
        if "blank_calibration_failure" in self.faults:
            return self._fail(
                start,
                device="plate_reader",
                action="blank_plate",
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
            self._state["plate_reader"].update(
                {
                    "status": "LOADING",
                    "door_state": "OPEN",
                    "tray_state": "EXTENDED",
                    "wavelength_nm": float(wavelength_nm),
                    "blanked": False,
                }
            )
            self._state["device_interlocks"].update(
                {"plate_reader_door": "OPEN", "plate_reader_tray": "EXTENDED"}
            )
        self._emit(
            device="plate_reader",
            action="open_door",
            message="Opening plate reader and accepting the measurement plate",
            progress=0.15,
            metadata={"step": "BlankSpectrophotometer"},
        )
        self._pause()
        with self._lock:
            self._state["plate_reader"].update(
                {"status": "RUNNING", "door_state": "CLOSED", "tray_state": "LOADED"}
            )
            self._state["device_interlocks"].update(
                {"plate_reader_door": "CLOSED", "plate_reader_tray": "LOADED"}
            )
            self._state["measurement_plate"]["status"] = "READING_BLANK"
        self._emit(
            device="plate_reader",
            action="blank_plate",
            message=f"Blanking plate well A2 at {wavelength_nm:.0f} nm",
            progress=0.55,
        )
        self._pause()
        with self._lock:
            self._state["spectrophotometer"].update(
                {"status": "READY", "blanked": True}
            )
            self._state["plate_reader"].update(
                {"status": "READY", "blanked": True, "door_state": "CLOSED"}
            )
            self._state["measurement_plate"]["status"] = "BLANKED"
        self._emit(
            device="plate_reader",
            action="blank_plate",
            message="Blank calibration completed",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="plate_reader",
            action="blank_plate",
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
            blanked = bool(self._state["plate_reader"]["blanked"])
        if not blanked:
            return self._fail(
                start,
                device="plate_reader",
                action="measure_plate_absorbance",
                message="Absorbance measurement requires a valid blank",
                error_code="SPECTROPHOTOMETER_NOT_BLANKED",
            )
        if "invalid_absorbance" in self.faults:
            return self._fail(
                start,
                device="plate_reader",
                action="measure_plate_absorbance",
                message="Spectrophotometer returned an invalid absorbance value",
                error_code="INVALID_ABSORBANCE",
            )
        with self._lock:
            self._state["phase"] = "MEASURING_OD"
            self._state["spectrophotometer"]["status"] = "RUNNING"
            self._state["plate_reader"].update({"status": "RUNNING", "tray_state": "LOADED"})
            self._state["measurement_plate"]["status"] = "READING_SAMPLE"
        self._emit(
            device="plate_reader",
            action="measure_plate_absorbance",
            message=f"Measuring plate well A1 at {wavelength_nm:.0f} nm",
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
            self._state["plate_reader"].update(
                {
                    "status": "COMPLETED",
                    "wavelength_nm": float(wavelength_nm),
                    "absorbance": float(expected_absorbance),
                    "tray_state": "LOADED",
                }
            )
            self._state["measurement_plate"].update(
                {"status": "MEASURED", "absorbance": float(expected_absorbance)}
            )
        self._emit(
            device="plate_reader",
            action="measure_plate_absorbance",
            message=f"Absorbance measured: {expected_absorbance:.3f}",
            status="completed",
            progress=1.0,
        )
        return self._result(
            start,
            ok=True,
            device="plate_reader",
            action="measure_plate_absorbance",
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
            self._state["device_interlocks"]["media_valve"] = "OPEN"
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
            if "pump_a_blocked" in self.faults and index == 3:
                with self._lock:
                    self._state["media_pump"]["status"] = "DECELERATING"
                    self._state["liquid_handler"]["status"] = "FAULT"
                    self._state["device_interlocks"]["media_valve"] = "CLOSED"
                return self._fail(
                    start,
                    device="media_pump",
                    action="dispense_medium",
                    message=f"Media pump blocked after {self._state['target_reactor']['volume_ml']:.1f} mL was delivered",
                    error_code="MEDIA_PUMP_BLOCKED",
                )
            if "aspirate_dispense_failure" in self.faults and index == 4:
                with self._lock:
                    self._state["media_pump"]["status"] = "OFF"
                    self._state["liquid_handler"]["status"] = "FAULT"
                    self._state["device_interlocks"]["media_valve"] = "CLOSED"
                return self._fail(
                    start,
                    device="liquid_handler",
                    action="dispense_medium",
                    message="Liquid handler stopped after a dispense error",
                    error_code="ASPIRATE_DISPENSE_FAILURE",
                )
        with self._lock:
            self._state["media_pump"]["status"] = "OFF"
            self._state["device_interlocks"]["media_valve"] = "CLOSED"
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
            self._state["device_interlocks"]["incubator_door"] = "CLOSED"
            self._set_attachment("target_flask", "incubator.slot_1", "incubator")
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
            self._state["plate_reader"].update(
                {"status": "STANDBY", "door_state": "CLOSED", "tray_state": "HOME"}
            )
            self._state["robot_arm"].update(
                {"status": "STANDBY", "pose": "HOME", "gripper_state": "OPEN"}
            )
            self._state["mobile_robot"].update(
                {"status": "STANDBY", "location": "home", "pose": "HOME", "gripper_state": "OPEN"}
            )
            self._state["transport"].update({"motion_phase": "COMPLETED", "progress": 1.0})
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
            self._state["plate_reader"].update(
                {"status": "SAFE", "door_state": "CLOSED", "tray_state": "HOME"}
            )
            self._state["robot_arm"].update(
                {"status": "DECELERATING", "pose": "SAFE_STOP"}
            )
            self._state["mobile_robot"].update(
                {"status": "DECELERATING", "pose": "SAFE_STOP"}
            )
            self._state["transport"].update({"motion_phase": "CONTROLLED_STOP"})
            self._state["device_interlocks"].update(
                {
                    "plate_reader_door": "CLOSED",
                    "plate_reader_tray": "HOME",
                    "media_valve": "CLOSED",
                }
            )
            if self._state["incubator"]["status"] != "RUNNING":
                self._state["incubator"]["status"] = "STANDBY"
        self._emit(
            device="safety_controller",
            action="safe_shutdown",
            message=f"Controlled stop started: {reason}",
            progress=0.35,
            metadata={"step": "SafeShutdown"},
        )
        self._pause()
        with self._lock:
            arm_holds_labware = any(
                parent == "robot_arm.gripper" for parent in self._state["attachments"].values()
            )
            self._state["robot_arm"].update(
                {
                    "status": "SAFE_HOLD" if arm_holds_labware else "SAFE",
                    "pose": "HOLD" if arm_holds_labware else "HOME",
                    "gripper_state": "CLOSED" if arm_holds_labware else "OPEN",
                }
            )
            self._state["mobile_robot"].update(
                {"status": "SAFE", "pose": "HOME", "gripper_state": "OPEN"}
            )
            self._state["transport"].update({"motion_phase": "CANCELLED"})
        self._emit(
            device="robot_arm",
            action="return_home",
            message="Robot retracted vertically and reached its safe pose",
            status="completed",
            progress=0.82,
            metadata={"step": "SafeShutdown"},
        )
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
