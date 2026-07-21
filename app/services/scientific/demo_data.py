from __future__ import annotations

import csv
import io
import math
from pathlib import Path


DEMO_XLSX = Path("data/raw/experiment_data/experiment_data__algae_growth_curve__v1__zh.xlsx")


def scientific_demo_file() -> tuple[bytes, str, dict[str, object]]:
    """Return the laboratory workbook when present, otherwise a CI-safe 25-condition fixture."""
    if DEMO_XLSX.exists():
        return DEMO_XLSX.read_bytes(), DEMO_XLSX.name, {
            "metric_name": "biomass",
            "response_unit": "g/L",
        }
    return synthetic_growth_csv(), "builtin_algae_growth_25.csv", {
        "time_column": "time_h",
        "response_column": "biomass",
        "factor_columns": ["light", "temperature", "nitrogen", "tf"],
        "response_unit": "g/L",
        "metric_name": "biomass",
    }


def synthetic_growth_csv() -> bytes:
    """Generate 25 deterministic four-factor curves without scientific-code mocking."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["time_h", "light", "temperature", "nitrogen", "tf", "biomass"])
    for index in range(25):
        light = (10.0, 20.0, 30.0, 40.0, 50.0)[index % 5]
        temperature = (20.0, 24.0, 28.0, 32.0, 36.0)[index // 5]
        nitrogen = (0.5, 1.0, 1.5)[index % 3]
        tf = (0.8, 1.0, 1.2)[(index // 3) % 3]
        rate = 0.018 - 0.000055 * (light - 35.0) ** 2 - 0.00013 * (temperature - 28.0) ** 2
        rate += 0.0015 * nitrogen + 0.0008 * tf
        carrying = 1.2 + 0.012 * light + 0.18 * nitrogen - 0.025 * abs(temperature - 28.0)
        for point in range(11):
            time_h = point * 12.0
            biomass = carrying / (1.0 + ((carrying / 0.08) - 1.0) * math.exp(-max(rate, 0.002) * time_h))
            writer.writerow([time_h, light, temperature, nitrogen, tf, round(biomass, 8)])
    return buffer.getvalue().encode("utf-8")
