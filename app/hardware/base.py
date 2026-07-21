from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class HardwareResult:
    ok: bool
    device: str
    action: str
    message: str
    state: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None


class SpectrophotometerController(Protocol):
    """Device contract for absorbance measurements."""

    def blank(self, wavelength_nm: float) -> HardwareResult: ...

    def measure_absorbance(
        self,
        wavelength_nm: float,
        expected_absorbance: float,
    ) -> HardwareResult: ...


class LiquidHandlerController(Protocol):
    """Device contract for the fixed-volume subculture recipe."""

    def check_materials(self, expected_media_ml: float) -> HardwareResult: ...

    def dispense_medium(self, target: str, volume_ml: float) -> HardwareResult: ...

    def transfer_seed(
        self,
        source: str,
        target: str,
        volume_ml: float,
    ) -> HardwareResult: ...

    def mix_and_seal(self, reactor: str) -> HardwareResult: ...


class IncubatorController(Protocol):
    """Device contract for the light incubator."""

    def configure(self, temperature_c: float, light_lux: float) -> HardwareResult: ...


class SubcultureWorkcell(Protocol):
    """Aggregate contract used by the workflow and replaceable backends."""

    spectrophotometer: SpectrophotometerController
    liquid_handler: LiquidHandlerController
    incubator_controller: IncubatorController

    def snapshot(self) -> dict[str, Any]: ...

    def relocate_sample(
        self,
        *,
        task_id: str,
        location: str,
        container: str,
    ) -> HardwareResult: ...

    def cleanup(self) -> HardwareResult: ...

    def safe_shutdown(self, reason: str) -> HardwareResult: ...


class HardwareController(SubcultureWorkcell, Protocol):
    """Contract shared by simulated and real laboratory hardware adapters."""

    def snapshot(self) -> dict[str, Any]: ...

    def sterilize_workspace(self, duration_min: float) -> HardwareResult: ...

    def check_materials(self, expected_media_ml: float) -> HardwareResult: ...

    def dispense_medium(self, target: str, volume_ml: float) -> HardwareResult: ...

    def transfer_seed(
        self,
        source: str,
        target: str,
        volume_ml: float,
    ) -> HardwareResult: ...

    def close_reactor(self, reactor: str) -> HardwareResult: ...

    def configure_incubator(
        self,
        temperature_c: float,
        light_lux: float,
    ) -> HardwareResult: ...

    def cleanup(self) -> HardwareResult: ...

    def safe_shutdown(self, reason: str) -> HardwareResult: ...
