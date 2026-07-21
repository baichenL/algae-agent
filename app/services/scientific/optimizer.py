from __future__ import annotations

import hashlib
import itertools
from typing import Any

import numpy as np

from app.services.scientific.models import ExperimentCondition


def _factor_names(metrics: list[dict[str, Any]]) -> list[str]:
    common: set[str] | None = None
    for item in metrics:
        keys = {str(key) for key, value in (item.get("condition") or {}).items() if isinstance(value, (int, float))}
        common = keys if common is None else common & keys
    return sorted(common or [])


def _feature_mode(sample_count: int, factor_count: int) -> str:
    full_terms = 1 + 2 * factor_count + factor_count * (factor_count - 1) // 2
    if sample_count >= max(8, full_terms + 2):
        return "full_quadratic_interactions"
    if sample_count >= 2 * factor_count + 3:
        return "quadratic_main_effects"
    if sample_count >= factor_count + 3:
        return "linear"
    return "insufficient_design"


def _features(x: np.ndarray, mode: str) -> np.ndarray:
    columns = [np.ones(len(x))]
    columns.extend(x[:, index] for index in range(x.shape[1]))
    if mode in {"quadratic_main_effects", "full_quadratic_interactions"}:
        columns.extend(x[:, index] ** 2 for index in range(x.shape[1]))
    if mode == "full_quadratic_interactions":
        columns.extend(
            x[:, left] * x[:, right]
            for left in range(x.shape[1])
            for right in range(left + 1, x.shape[1])
        )
    return np.column_stack(columns)


def _ridge_fit(design: np.ndarray, y: np.ndarray, ridge_lambda: float = 1e-3) -> np.ndarray:
    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y


def fit_response_surface(metrics: list[dict[str, Any]], *, target_key: str = "max_value") -> dict[str, Any]:
    usable = [item for item in metrics if item.get(target_key) is not None]
    names = _factor_names(usable)
    if not names:
        return {"status": "insufficient_design", "reason": "no_numeric_common_factors", "factor_names": []}
    x = np.array([[float(item["condition"][name]) for name in names] for item in usable], dtype=float)
    y = np.array([float(item[target_key]) for item in usable], dtype=float)
    mode = _feature_mode(len(usable), len(names))
    if mode == "insufficient_design":
        return {
            "status": mode, "reason": "not_enough_independent_conditions", "factor_names": names,
            "sample_count": len(usable),
        }
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    design = _features(standardized, mode)
    coefficients = _ridge_fit(design, y)
    predicted = design @ coefficients
    residual_values = y - predicted
    ss_res = float(np.sum(residual_values ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    importances = []
    linear = coefficients[1:1 + len(names)]
    denominator = float(np.sum(np.abs(linear))) or 1.0
    for name, coefficient in zip(names, linear):
        importances.append({"factor": name, "coefficient": float(coefficient), "importance": abs(float(coefficient)) / denominator})
    importances.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "status": "ok",
        "mode": mode,
        "factor_names": names,
        "sample_count": len(usable),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "r_squared": r_squared,
        "factor_importance": importances,
        "residuals": {item["batch_id"]: float(value) for item, value in zip(usable, residual_values)},
        "observed_conditions": [item["condition"] for item in usable],
        "observed_values": y.tolist(),
    }


def _predict(model: dict[str, Any], rows: np.ndarray, coefficients: np.ndarray | None = None) -> np.ndarray:
    x = (rows - np.array(model["mean"], dtype=float)) / np.array(model["scale"], dtype=float)
    design = _features(x, model["mode"])
    beta = coefficients if coefficients is not None else np.array(model["coefficients"], dtype=float)
    return design @ beta


def propose_conditions(
    model: dict[str, Any],
    *,
    direction: str,
    run_seed: str,
    candidate_count: int = 4,
    bootstrap_count: int = 200,
    excluded_conditions: list[dict[str, float]] | None = None,
    candidate_pool: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    if model.get("status") != "ok":
        return {"status": "insufficient_design", "conditions": [], "model": model}
    names = list(model["factor_names"])
    observed = [tuple(float(item[name]) for name in names) for item in model["observed_conditions"]]
    excluded = {
        tuple(float(item[name]) for name in names)
        for item in (excluded_conditions or [])
        if all(name in item for name in names)
    }
    if candidate_pool:
        grid = [
            tuple(float(item[name]) for name in names)
            for item in candidate_pool
            if all(name in item for name in names)
        ]
        grid = list(dict.fromkeys(row for row in grid if row not in set(observed) and row not in excluded))
    else:
        levels = [sorted({row[index] for row in observed}) for index in range(len(names))]
        grid = [row for row in itertools.product(*levels) if row not in set(observed) and row not in excluded]
    if not grid:
        return {"status": "candidate_space_exhausted", "conditions": [], "model": model}
    candidate_array = np.array(grid, dtype=float)
    predictions = _predict(model, candidate_array)

    observed_x = np.array(observed, dtype=float)
    observed_y = np.array(model["observed_values"], dtype=float)
    standardized = (observed_x - np.array(model["mean"])) / np.array(model["scale"])
    design = _features(standardized, model["mode"])
    seed = int(hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    bootstrap_predictions = []
    for _ in range(bootstrap_count):
        indices = rng.integers(0, len(observed_y), len(observed_y))
        beta = _ridge_fit(design[indices], observed_y[indices])
        bootstrap_predictions.append(_predict(model, candidate_array, beta))
    uncertainty = np.std(np.vstack(bootstrap_predictions), axis=0)
    nearest_distance = np.array([
        min(float(np.linalg.norm((row - previous) / np.array(model["scale"]))) for previous in observed_x)
        for row in candidate_array
    ])

    sign = 1.0 if direction == "maximize" else -1.0
    improvement = sign * predictions
    def normalize(values: np.ndarray) -> np.ndarray:
        span = float(np.max(values) - np.min(values))
        return (values - np.min(values)) / span if span > 1e-12 else np.zeros_like(values)
    acquisition = normalize(improvement) + 0.5 * normalize(uncertainty) + 0.1 * normalize(nearest_distance)
    ordered = list(np.argsort(-acquisition))
    selected: list[int] = []
    for index in ordered:
        if not selected or all(np.linalg.norm(candidate_array[index] - candidate_array[other]) > 1e-9 for other in selected):
            selected.append(int(index))
        if len(selected) >= candidate_count:
            break
    best_observed_index = int(np.argmax(observed_y) if direction == "maximize" else np.argmin(observed_y))
    conditions = [
        ExperimentCondition(
            condition_id=f"candidate_{position + 1}",
            factors={name: float(candidate_array[index, offset]) for offset, name in enumerate(names)},
            predicted_value=float(predictions[index]), uncertainty=float(uncertainty[index]),
            acquisition_score=float(acquisition[index]), role="candidate",
        )
        for position, index in enumerate(selected)
    ]
    control_factors = {name: float(observed_x[best_observed_index, offset]) for offset, name in enumerate(names)}
    conditions.append(
        ExperimentCondition(
            "control_best_observed", control_factors, float(observed_y[best_observed_index]), 0.0, 0.0, "control"
        )
    )
    return {"status": "ok", "conditions": conditions, "model": model}
