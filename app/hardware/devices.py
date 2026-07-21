from __future__ import annotations

from typing import TYPE_CHECKING

from app.hardware.base import HardwareResult

if TYPE_CHECKING:
    from app.hardware.simulator import SimulatedHardware


class SimulatedSpectrophotometer:
    def __init__(self, workcell: "SimulatedHardware") -> None:
        self._workcell = workcell

    def blank(self, wavelength_nm: float) -> HardwareResult:
        return self._workcell.blank_spectrophotometer(wavelength_nm)

    def measure_absorbance(
        self,
        wavelength_nm: float,
        expected_absorbance: float,
    ) -> HardwareResult:
        return self._workcell.measure_absorbance(
            wavelength_nm=wavelength_nm,
            expected_absorbance=expected_absorbance,
        )


class SimulatedLiquidHandler:
    def __init__(self, workcell: "SimulatedHardware") -> None:
        self._workcell = workcell

    def check_materials(self, expected_media_ml: float) -> HardwareResult:
        return self._workcell.check_materials(expected_media_ml)

    def dispense_medium(self, target: str, volume_ml: float) -> HardwareResult:
        return self._workcell.dispense_medium(target, volume_ml)

    def transfer_seed(
        self,
        source: str,
        target: str,
        volume_ml: float,
    ) -> HardwareResult:
        return self._workcell.transfer_seed(source, target, volume_ml)

    def mix_and_seal(self, reactor: str) -> HardwareResult:
        return self._workcell.mix_and_seal(reactor)


class SimulatedIncubator:
    def __init__(self, workcell: "SimulatedHardware") -> None:
        self._workcell = workcell

    def configure(self, temperature_c: float, light_lux: float) -> HardwareResult:
        return self._workcell.configure_incubator(temperature_c, light_lux)
