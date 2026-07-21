from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

import numpy as np

from app.services.scientific.models import DiagnosisHypothesis, DiagnosisReport


def assess_data_quality(dataset: dict[str, Any], metric_name: str) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    batch_reports: list[dict[str, Any]] = []
    for batch in dataset.get("batches") or []:
        measurements = [m for m in batch.get("measurements") or [] if m.get("metric_name") == metric_name]
        times = [float(item["elapsed_hours"]) for item in measurements]
        values = [float(item["value"]) for item in measurements]
        original = sorted(measurements, key=lambda item: int(item.get("source_row") or 0))
        original_times = [float(item["elapsed_hours"]) for item in original]
        batch_flags: list[str] = []
        if len(measurements) < 5:
            batch_flags.append("insufficient_timepoints")
        duplicates = [time for time, count in Counter(times).items() if count > 1]
        if duplicates:
            batch_flags.append("duplicate_timepoints")
        if any(right < left for left, right in zip(original_times, original_times[1:])):
            batch_flags.append("non_monotonic_source_time")
        if any(value <= 0 for value in values):
            batch_flags.append("non_positive_growth_value")
        if len(times) >= 3:
            gaps = [right - left for left, right in zip(sorted(set(times)), sorted(set(times))[1:])]
            if gaps and max(gaps) > 2.5 * statistics.median(gaps):
                batch_flags.append("irregular_sampling_gap")
        for code in batch_flags:
            flags.append({"batch_id": batch["id"], "code": code})
        batch_reports.append(
            {
                "batch_id": batch["id"],
                "timepoint_count": len(measurements),
                "flags": batch_flags,
                "passed": not batch_flags,
            }
        )
    return {
        "passed": not any(item["code"] in {"insufficient_timepoints", "non_positive_growth_value"} for item in flags),
        "batch_count": len(batch_reports),
        "flag_count": len(flags),
        "flags": flags,
        "batches": batch_reports,
    }


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(coefficients[1]), float(coefficients[0]), r_squared


def compute_growth_metrics(dataset: dict[str, Any], metric_name: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for batch in dataset.get("batches") or []:
        measurements = sorted(
            [m for m in batch.get("measurements") or [] if m.get("metric_name") == metric_name],
            key=lambda item: float(item["elapsed_hours"]),
        )
        times = np.array([float(item["elapsed_hours"]) for item in measurements], dtype=float)
        values = np.array([float(item["value"]) for item in measurements], dtype=float)
        flags: list[str] = []
        if len(times) < 3 or np.any(values <= 0):
            reports.append(
                {
                    "batch_id": batch["id"], "condition": batch.get("condition") or {},
                    "metric_name": metric_name, "status": "insufficient_data", "flags": ["growth_rate_unavailable"],
                }
            )
            continue
        candidates: list[dict[str, Any]] = []
        upper_window = min(6, len(times))
        log_values = np.log(values)
        for width in range(3, upper_window + 1):
            for start in range(0, len(times) - width + 1):
                end = start + width
                slope, _, r_squared = _linear_fit(times[start:end], log_values[start:end])
                if slope > 0:
                    candidates.append(
                        {"slope": slope, "r_squared": r_squared, "start": start, "end": end, "width": width}
                    )
        eligible = [item for item in candidates if item["r_squared"] >= 0.80]
        pool = eligible or candidates
        if not pool:
            max_growth_rate = None
            doubling_time = None
            lag_end = None
            fit_r_squared = None
            flags.append("no_positive_log_linear_window")
        else:
            best = max(pool, key=lambda item: (item["slope"], item["r_squared"], item["width"]))
            max_growth_rate = best["slope"]
            doubling_time = math.log(2.0) / max_growth_rate
            lag_end = float(times[best["start"]])
            fit_r_squared = best["r_squared"]
            if fit_r_squared < 0.80:
                flags.append("weak_log_linear_fit")
        reports.append(
            {
                "batch_id": batch["id"],
                "condition": batch.get("condition") or {},
                "metric_name": metric_name,
                "status": "ok" if max_growth_rate is not None else "degraded",
                "max_value": float(np.max(values)),
                "final_value": float(values[-1]),
                "auc": float(np.trapezoid(values, times)),
                "max_specific_growth_rate": max_growth_rate,
                "doubling_time_hours": doubling_time,
                "lag_end_hours": lag_end,
                "fit_r_squared": fit_r_squared,
                "timepoint_count": len(times),
                "flags": flags,
            }
        )
    return reports


def detect_anomalies(metrics: list[dict[str, Any]], residuals: dict[str, float] | None = None) -> list[dict[str, Any]]:
    usable = [item for item in metrics if item.get("status") != "insufficient_data" and item.get("max_value") is not None]
    if not usable:
        return []
    residuals = residuals or {}
    values = [float(residuals.get(item["batch_id"], item["max_value"])) for item in usable]
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    fallback_std = statistics.pstdev(values) if len(values) > 1 else 0.0
    anomalies: list[dict[str, Any]] = []
    for item, value in zip(usable, values):
        if mad > 1e-12:
            score = 0.6745 * (value - median) / mad
        elif fallback_std > 1e-12:
            score = (value - median) / fallback_std
        else:
            score = 0.0
        if abs(score) >= 3.5:
            anomalies.append(
                {
                    "batch_id": item["batch_id"],
                    "kind": "model_residual" if item["batch_id"] in residuals else "robust_metric_outlier",
                    "score": float(score),
                    "direction": "lower_than_expected" if score < 0 else "higher_than_expected",
                    "observed_metric": item.get("max_value"),
                    "condition": item.get("condition") or {},
                }
            )
    return sorted(anomalies, key=lambda item: abs(item["score"]), reverse=True)


def build_diagnosis(
    *,
    anomalies: list[dict[str, Any]],
    quality: dict[str, Any],
    factor_importance: list[dict[str, Any]],
    evidence: list[dict[str, Any]] | None = None,
) -> DiagnosisReport:
    evidence = evidence or []
    evidence_score = min(1.0, len(evidence) / 2.0)
    hypotheses: list[DiagnosisHypothesis] = []
    if quality.get("flag_count"):
        signal = min(1.0, quality["flag_count"] / max(1, quality.get("batch_count", 1)))
        score = 0.5 * signal + 0.3 * 0.0 + 0.2 * 0.5
        hypotheses.append(
            DiagnosisHypothesis(
                "data_quality", "measurement_or_sampling_quality", score, "observed_association",
                signal, 0.0, 0.5, (), ("Resolve data-quality flags before biological attribution.",),
            )
        )
    for index, factor in enumerate(factor_importance[:4], start=1):
        data_signal = float(factor.get("importance") or 0.0)
        score = 0.5 * data_signal + 0.3 * evidence_score + 0.2 * 0.25
        hypotheses.append(
            DiagnosisHypothesis(
                f"factor_{index}", f"factor_association:{factor['factor']}", score,
                "literature_supported" if evidence else "needs_experiment",
                data_signal, evidence_score, 0.25, tuple(evidence[:3]),
                ("Association is not proof of causality; validate with a controlled experiment.",),
            )
        )
    hypotheses.sort(key=lambda item: item.score, reverse=True)
    if anomalies:
        conclusion = "检测到偏离模型预期的培养批次；以下条目仅为需要后续对照实验验证的候选原因。"
    else:
        conclusion = "当前数据未达到预设异常阈值；仍可根据因素关联设计下一轮信息增益实验。"
    return DiagnosisReport(tuple(anomalies), tuple(hypotheses), conclusion, causal_claim=False)

