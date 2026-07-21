from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from app.services.scientific.models import ExperimentDesignSpec, canonical_hash


class LabCapabilityAdapter(Protocol):
    def descriptor(self) -> dict[str, Any]: ...
    def capabilities(self) -> list[str]: ...
    def snapshot(self) -> dict[str, Any]: ...
    def validate_design(self, design: ExperimentDesignSpec) -> dict[str, Any]: ...
    def simulate_design(self, design: ExperimentDesignSpec, *, dataset: dict[str, Any]) -> dict[str, Any]: ...
    def compile_protocol(self, design: ExperimentDesignSpec) -> dict[str, Any]: ...


def _condition_match(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-9) -> bool:
    if set(left) != set(right):
        return False
    return all(abs(float(left[key]) - float(right[key])) <= tolerance for key in left)


@dataclass
class PredictiveSimulationAdapter:
    adapter_id: str = "sim-algae-lab-v1"
    max_culture_vessels: int = 12

    def descriptor(self) -> dict[str, Any]:
        return {
            "id": self.adapter_id,
            "name": "Predictive algae lab digital twin",
            "mode": "simulation_only",
            "provenance": "model_predicted",
        }

    def capabilities(self) -> list[str]:
        return ["prepare_conditions", "inoculate", "incubate", "measure_growth", "record_results"]

    def snapshot(self) -> dict[str, Any]:
        return {"status": "online", "mode": "simulation_only", "max_culture_vessels": self.max_culture_vessels}

    def validate_design(self, design: ExperimentDesignSpec) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        required_units = len(design.conditions) * int(design.replicates)
        if required_units > self.max_culture_vessels:
            issues.append(
                {
                    "code": "CAPACITY_EXCEEDED",
                    "message": f"Design requires {required_units} culture vessels but only {self.max_culture_vessels} are available.",
                    "required": required_units,
                    "available": self.max_culture_vessels,
                }
            )
        missing = sorted(set(design.required_capabilities) - set(self.capabilities()))
        if missing:
            issues.append({"code": "CAPABILITY_MISSING", "message": "Required capabilities are unavailable.", "missing": missing})
        if not any(item.role == "control" for item in design.conditions):
            issues.append({"code": "CONTROL_MISSING", "message": "A control condition is required."})
        return {
            "valid": not issues,
            "issues": issues,
            "required_culture_vessels": required_units,
            "available_culture_vessels": self.max_culture_vessels,
            "adapter": self.descriptor(),
        }

    def simulate_design(self, design: ExperimentDesignSpec, *, dataset: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate_design(design)
        if not validation["valid"]:
            return {"status": "failed", "validation": validation, "simulation_only": True}
        measurements = []
        for condition in design.conditions:
            start = max(0.02, min(0.2, condition.predicted_value * 0.25))
            final = max(start, condition.predicted_value)
            horizon = max(design.sampling_hours) if design.sampling_hours else 120.0
            for replicate in range(1, design.replicates + 1):
                for hour in design.sampling_hours:
                    progress = 0.0 if horizon <= 0 else float(hour) / float(horizon)
                    value = start + (final - start) * (1.0 - math.exp(-3.2 * progress)) / (1.0 - math.exp(-3.2))
                    measurements.append(
                        {
                            "condition_id": condition.condition_id,
                            "factors": condition.factors,
                            "elapsed_hours": float(hour),
                            "metric_name": design.target_metric,
                            "value": float(value),
                            "replicate_index": replicate,
                            "simulation_only": True,
                            "provenance": "model_predicted",
                        }
                    )
        return {
            "status": "success",
            "simulation_id": f"sim_{uuid.uuid4().hex[:16]}",
            "simulation_only": True,
            "provenance": "model_predicted",
            "measurements": measurements,
            "validation": validation,
        }

    def compile_protocol(self, design: ExperimentDesignSpec) -> dict[str, Any]:
        protocol = {
            "protocol_type": "diagnostic_screening",
            "protocol_version": 1,
            "design_hash": design.design_hash,
            "simulation_only": True,
            "steps": [
                {"order": 1, "operation": "prepare_conditions"},
                {"order": 2, "operation": "inoculate"},
                {"order": 3, "operation": "incubate", "sampling_hours": design.sampling_hours},
                {"order": 4, "operation": "measure_growth", "metric": design.target_metric},
                {"order": 5, "operation": "record_results"},
            ],
        }
        protocol["protocol_hash"] = canonical_hash(protocol)
        return protocol


@dataclass
class OfflineReplayLabAdapter(PredictiveSimulationAdapter):
    adapter_id: str = "offline-replay-v1"

    def descriptor(self) -> dict[str, Any]:
        return {
            "id": self.adapter_id,
            "name": "Held-out experimental data replay adapter",
            "mode": "simulation_only",
            "provenance": "offline_replay",
        }

    def simulate_design(self, design: ExperimentDesignSpec, *, dataset: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate_design(design)
        if not validation["valid"]:
            return {"status": "failed", "validation": validation, "simulation_only": True}
        hidden_ids = set((dataset.get("replay_split") or {}).get("hidden_batch_ids") or [])
        hidden = [item for item in dataset.get("batches") or [] if item.get("id") in hidden_ids]
        measurements = []
        matched = []
        for condition in design.conditions:
            source = next((batch for batch in hidden if _condition_match(batch.get("condition") or {}, condition.factors)), None)
            if not source:
                continue
            matched.append(source["id"])
            for item in source.get("measurements") or []:
                if item.get("metric_name") != design.target_metric:
                    continue
                measurements.append(
                    {
                        "condition_id": condition.condition_id,
                        "source_batch_id": source["id"],
                        "factors": condition.factors,
                        "elapsed_hours": float(item["elapsed_hours"]),
                        "metric_name": design.target_metric,
                        "value": float(item["value"]),
                        "replicate_index": int(item.get("replicate_index") or 1),
                        "simulation_only": True,
                        "provenance": "offline_replay",
                    }
                )
        if not measurements:
            fallback = super().simulate_design(design, dataset=dataset)
            fallback["provenance"] = "model_predicted"
            fallback["replay_fallback_reason"] = "no_exact_held_out_condition"
            return fallback
        return {
            "status": "success",
            "simulation_id": f"replay_{uuid.uuid4().hex[:16]}",
            "simulation_only": True,
            "provenance": "offline_replay",
            "measurements": measurements,
            "matched_hidden_batch_ids": matched,
            "validation": validation,
        }


def get_adapter(adapter_id: str) -> LabCapabilityAdapter:
    if adapter_id == "offline-replay-v1":
        return OfflineReplayLabAdapter()
    return PredictiveSimulationAdapter(adapter_id=adapter_id or "sim-algae-lab-v1")

