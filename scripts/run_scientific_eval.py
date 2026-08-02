from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.db import scientific as scientific_db
from app.services.scientific.adapters import PredictiveSimulationAdapter
from app.services.scientific.analysis import assess_data_quality, build_diagnosis, compute_growth_metrics
from app.services.scientific.demo_data import scientific_demo_file
from app.services.scientific.importer import parse_scientific_dataset
from app.services.scientific.models import ExperimentCondition, ExperimentDesignSpec
from app.services.scientific.optimizer import fit_response_surface, propose_conditions
from app.services.scientific.service import approve_and_simulate_proposal, run_scientific_task
from app.services.rag.ingestion.index_store import ingest_file
from scripts.run_agent_eval import configure_temp_db


def _record(cases: list[dict[str, Any]], case_id: str, passed: bool, **details: Any) -> None:
    cases.append({"id": case_id, "passed": bool(passed), "details": details})


def _small_dataset() -> dict[str, Any]:
    batches = []
    for batch_index in range(6):
        measurements = []
        for point in range(7):
            hour = float(point * 12)
            measurements.append({
                "elapsed_hours": hour,
                "metric_name": "biomass",
                "value": float(0.1 * np.exp((0.01 + batch_index * 0.0005) * hour)),
                "replicate_index": 1,
                "source_row": batch_index * 7 + point + 2,
            })
        batches.append({"id": f"b{batch_index}", "condition": {"x": float(batch_index)}, "measurements": measurements})
    return {"mapping": {"metric_name": "biomass"}, "batches": batches}


def _condition_key(condition: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(key), round(float(value), 10)) for key, value in condition.items()))


def run_eval(*, run_seed: int = 2025, work_dir: Path | None = None) -> dict[str, Any]:
    # This suite is explicitly offline and must not inherit a developer's
    # online embedding provider or network availability.
    os.environ["RAG_EMBEDDING_ENABLED"] = "true"
    os.environ["RAG_EMBEDDING_PROVIDER"] = "fake"
    os.environ["RAG_EMBEDDING_MODEL"] = "fake-scientific-eval-v1"
    os.environ["SCIENTIFIC_EVIDENCE_OFFLINE_FTS"] = "true"
    cases: list[dict[str, Any]] = []
    quality_expected: list[bool] = []
    quality_predicted: list[bool] = []
    growth_expected: list[float] = []
    growth_predicted: list[float] = []
    diagnosis_ranks: list[int | None] = []

    # Data quality fault injection (5 scenarios).
    for mutation, expected in (
        ("short", "insufficient_timepoints"),
        ("duplicate", "duplicate_timepoints"),
        ("nonpositive", "non_positive_growth_value"),
        ("gap", "irregular_sampling_gap"),
        ("reversed", "non_monotonic_source_time"),
    ):
        data = _small_dataset()
        values = data["batches"][0]["measurements"]
        if mutation == "short":
            del values[3:]
        elif mutation == "duplicate":
            values[1]["elapsed_hours"] = values[0]["elapsed_hours"]
        elif mutation == "nonpositive":
            values[2]["value"] = 0.0
        elif mutation == "gap":
            values[-1]["elapsed_hours"] = 500.0
        else:
            values[1]["elapsed_hours"] = -1.0
        flags = {item["code"] for item in assess_data_quality(data, "biomass")["flags"]}
        detected = expected in flags
        quality_expected.append(True)
        quality_predicted.append(detected)
        _record(cases, f"quality_{mutation}", detected, expected=expected, flags=sorted(flags))

    # Clean negatives are required to measure false positives.
    for index in range(3):
        flags = {
            item["code"]
            for item in assess_data_quality(_small_dataset(), "biomass")["flags"]
        }
        predicted_fault = bool(flags)
        quality_expected.append(False)
        quality_predicted.append(predicted_fault)
        _record(
            cases,
            f"quality_clean_{index + 1}",
            not predicted_fault,
            expected="clean",
            flags=sorted(flags),
        )

    # Deterministic metric and model-degradation checks (8 scenarios).
    for rate in (0.005, 0.01, 0.02, 0.03):
        data = _small_dataset()
        for item in data["batches"][0]["measurements"]:
            item["value"] = 0.1 * np.exp(rate * item["elapsed_hours"])
        actual = compute_growth_metrics(data, "biomass")[0]["max_specific_growth_rate"]
        growth_expected.append(rate)
        growth_predicted.append(actual)
        _record(cases, f"growth_rate_{rate}", abs(actual - rate) < 1e-8, expected=rate, actual=actual)
    for count, expected in ((3, "insufficient_design"), (4, "linear"), (6, "quadratic_main_effects"), (10, "full_quadratic_interactions")):
        metrics = [{"batch_id": f"b{i}", "condition": {"x": float(i)}, "max_value": float(i * i + 1)} for i in range(count)]
        model = fit_response_surface(metrics)
        actual = model.get("mode") or model.get("status")
        _record(cases, f"model_{count}", actual == expected, expected=expected, actual=actual)

    # Labeled diagnosis-boundary cases (4 scenarios).
    for label in ("light", "nitrogen", "temperature"):
        diagnosis = build_diagnosis(
            anomalies=[{"batch_id": "fault", "score": 3.0}],
            quality={"flag_count": 0, "batch_count": 1},
            factor_importance=[{"factor": label, "importance": 0.8}, {"factor": "other", "importance": 0.2}],
            evidence=[],
        ).to_dict()
        top3 = [item["candidate_cause"] for item in diagnosis["hypotheses"][:3]]
        target = f"factor_association:{label}"
        diagnosis_ranks.append(top3.index(target) + 1 if target in top3 else None)
        _record(cases, f"top3_{label}", target in top3 and not diagnosis["causal_claim"], top3=top3)
    quality_diagnosis = build_diagnosis(
        anomalies=[], quality={"flag_count": 2, "batch_count": 1}, factor_importance=[], evidence=[]
    ).to_dict()
    diagnosis_ranks.append(
        1
        if quality_diagnosis["hypotheses"][0]["candidate_cause"]
        == "measurement_or_sampling_quality"
        else None
    )
    _record(cases, "top3_sensor_quality", quality_diagnosis["hypotheses"][0]["candidate_cause"] == "measurement_or_sampling_quality")

    evidence_dir = work_dir or Path(tempfile.mkdtemp(prefix="algae_eval_evidence_"))
    evidence_path = evidence_dir / "paper__algae_growth_factors__2024__en.md"
    evidence_path.write_text(
        "Algae growth is associated with light, temperature, nitrogen availability, and culture conditions. "
        "These associations require controlled experiments and do not by themselves establish causality.",
        encoding="utf-8",
    )
    ingest_file(evidence_path)

    content, filename, mapping = scientific_demo_file()
    parsed = parse_scientific_dataset(content=content, filename=filename, strain_id="Chlorella_01", mapping=mapping)
    scientific_db.insert_dataset(parsed.dataset, parsed.batches)
    first = run_scientific_task(dataset_id=parsed.dataset["id"], offline_replay=True, run_seed=run_seed)
    second = run_scientific_task(dataset_id=parsed.dataset["id"], offline_replay=True, run_seed=run_seed)
    artifacts = first["artifacts"]
    artifact_types = [item["artifact_type"] for item in artifacts]
    required = ("goal_contract", "plan_graph", "observation", "verification_report", "plan_patch", "experiment_design")
    for artifact_type in required:
        _record(cases, f"artifact_{artifact_type}", artifact_type in artifact_types)

    design = [item["payload"] for item in artifacts if item["artifact_type"] == "experiment_design"][-1]
    design2 = [item["payload"] for item in second["artifacts"] if item["artifact_type"] == "experiment_design"][-1]
    patches = [item["payload"] for item in artifacts if item["artifact_type"] == "plan_patch"]
    _record(cases, "capacity_repaired", len(design["conditions"]) * design["replicates"] == 12, design_hash=design["design_hash"])
    _record(cases, "dynamic_plan_replace", bool(patches and patches[0]["replace_nodes"]), patch=patches[0] if patches else None)
    _record(cases, "seed_reproducible", design["design_hash"] == design2["design_hash"], first=design["design_hash"], second=design2["design_hash"])

    simulation = approve_and_simulate_proposal(first["proposal"]["pending_id"], reviewed_by="eval-approver")
    _record(cases, "simulation_provenance", simulation.get("simulation", {}).get("provenance") == "offline_replay")
    _record(cases, "simulation_only", simulation.get("simulation", {}).get("simulation_only") is True)
    _record(cases, "fresh_approval_each_cycle", bool(simulation.get("next_pending", {}).get("pending_id")))

    diagnosis = [item["payload"] for item in artifacts if item["artifact_type"] == "diagnosis_report"][-1]
    retained = [item for item in diagnosis.get("hypotheses") or [] if item.get("candidate_cause", "").startswith("factor_association:")]
    covered = [item for item in retained if item.get("evidence_refs")]
    citation_coverage = len(covered) / len(retained) if retained else 1.0
    _record(cases, "retained_hypothesis_citation_coverage", citation_coverage >= 0.9, coverage=citation_coverage)

    # Standard 17/8 offline replay over a known quadratic response surface.
    # Multiple fixed seeds prevent a favorable single split from gaming regret.
    benchmark = [
        {
            "batch_id": f"p_{left}_{right}",
            "condition": {"light_level": float(left), "temperature_level": float(right)},
            "max_value": 10.0 - (left - 3.2) ** 2 - 0.7 * (right - 1.7) ** 2 + 0.15 * left * right,
        }
        for left in range(5)
        for right in range(5)
    ]
    full_regrets: list[float] = []
    random_medians: list[float] = []
    for split_seed in (11, 23, 47, run_seed, 4099):
        ordered = sorted(
            benchmark,
            key=lambda item: __import__("hashlib").sha256(f"{split_seed}:{item['batch_id']}".encode()).hexdigest(),
        )
        visible, hidden = ordered[:17], ordered[17:]
        model = fit_response_surface(visible)
        proposals = propose_conditions(
            model, direction="maximize", run_seed=f"{split_seed}:0", candidate_count=4,
            candidate_pool=[item["condition"] for item in hidden],
        )["conditions"]
        truth = {_condition_key(item["condition"]): float(item["max_value"]) for item in hidden}
        selected_values = [truth[_condition_key(item.factors)] for item in proposals[:3] if _condition_key(item.factors) in truth]
        optimum = max(truth.values())
        full_regrets.append(optimum - max(selected_values))
        values = list(truth.values())
        per_seed_random = []
        for random_seed in range(40):
            rng = np.random.default_rng(random_seed)
            selected = rng.choice(len(values), size=3, replace=False)
            per_seed_random.append(optimum - max(values[int(index)] for index in selected))
        random_medians.append(float(np.median(per_seed_random)))
    full_regret = float(np.median(full_regrets))
    random_median = float(np.median(random_medians))
    regret_improvement = (random_median - full_regret) / random_median if random_median > 1e-12 else 0.0
    _record(
        cases, "offline_replay_simple_regret", regret_improvement >= 0.2,
        full_regrets=full_regrets, random_medians=random_medians,
        full_regret_median=full_regret, random_median=random_median, improvement=regret_improvement,
    )

    # Direct/no-replan keeps the original 15-unit design and fails the 12-vessel profile.
    oversized = ExperimentDesignSpec(
        "baseline", "baseline", "biomass", "maximize",
        [ExperimentCondition(f"c{i}", {"x": float(i)}, 1.0, 0.1, 0.5, "control" if i == 4 else "candidate") for i in range(5)],
        3, [0.0, 12.0], ["prepare_conditions", "inoculate", "incubate", "measure_growth", "record_results"],
    ).freeze()
    baseline_valid = PredictiveSimulationAdapter().validate_design(oversized)["valid"]
    full_recovery = all(next(item["passed"] for item in cases if item["id"] == name) for name in ("capacity_repaired", "dynamic_plan_replace", "fresh_approval_each_cycle"))
    no_replan_success = 1.0 if baseline_valid else 0.0
    full_success = 1.0 if full_recovery else 0.0
    constraint_satisfied = len(design["conditions"]) * design["replicates"] == 12

    passed = sum(1 for item in cases if item["passed"])
    top3_cases = [item for item in cases if item["id"].startswith("top3_")]
    quality_tp = sum(expected and predicted for expected, predicted in zip(quality_expected, quality_predicted))
    quality_fp = sum(not expected and predicted for expected, predicted in zip(quality_expected, quality_predicted))
    quality_fn = sum(expected and not predicted for expected, predicted in zip(quality_expected, quality_predicted))
    quality_precision = quality_tp / (quality_tp + quality_fp) if quality_tp + quality_fp else None
    quality_recall = quality_tp / (quality_tp + quality_fn) if quality_tp + quality_fn else None
    quality_f1 = (
        2 * quality_precision * quality_recall / (quality_precision + quality_recall)
        if quality_precision is not None and quality_recall is not None and quality_precision + quality_recall
        else None
    )
    growth_errors = [predicted - expected for expected, predicted in zip(growth_expected, growth_predicted)]
    report = {
        "scenario_count": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "pass_rate": passed / len(cases),
        "metrics": {
            "data_quality_precision": quality_precision,
            "data_quality_recall": quality_recall,
            "data_quality_f1": quality_f1,
            "data_quality_tp": quality_tp,
            "data_quality_fp": quality_fp,
            "data_quality_fn": quality_fn,
            "data_quality_labels": quality_expected,
            "data_quality_predictions": quality_predicted,
            "growth_rate_mae": float(np.mean(np.abs(growth_errors))),
            "growth_rate_rmse": float(np.sqrt(np.mean(np.square(growth_errors)))),
            "growth_rate_expected": growth_expected,
            "growth_rate_predicted": growth_predicted,
            "candidate_cause_hit_at_1": sum(rank == 1 for rank in diagnosis_ranks) / len(diagnosis_ranks),
            "candidate_cause_hit_at_3": sum(rank is not None and rank <= 3 for rank in diagnosis_ranks) / len(diagnosis_ranks),
            "candidate_cause_mrr": sum(1.0 / rank if rank else 0.0 for rank in diagnosis_ranks) / len(diagnosis_ranks),
            "candidate_cause_ranks": diagnosis_ranks,
            "top3_candidate_cause_hit_rate": sum(item["passed"] for item in top3_cases) / len(top3_cases),
            "constraint_satisfaction_rate": 1.0 if constraint_satisfied else 0.0,
            "constraint_violation_rate": 0.0 if constraint_satisfied else 1.0,
            "simulation_provenance_isolation_rate": 1.0 if simulation.get("simulation", {}).get("simulation_only") else 0.0,
            "plan_patch_correctness_rate": 1.0 if full_recovery else 0.0,
            "full_loop_failure_recovery_rate": full_success,
            "task_success_rate": full_success,
            "no_replan_failure_recovery_rate": no_replan_success,
            "recovery_gain_percentage_points": 100.0 * (full_success - no_replan_success),
            "citation_coverage": citation_coverage,
            "offline_replay_simple_regret": full_regret,
            "random_selection_simple_regret_median": random_median,
            "simple_regret_improvement": regret_improvement,
            "full_regrets_by_seed": full_regrets,
            "random_regrets_by_seed": random_medians,
        },
        "baselines": {
            "direct_single_pass": {"capacity_recovery": no_replan_success, "verification": False, "evidence_boundary": False},
            "no_replan": {"capacity_recovery": no_replan_success, "verification": True, "evidence_boundary": True},
            "no_verifier": {"capacity_recovery": no_replan_success, "verification": False, "evidence_boundary": True},
            "no_evidence": {"capacity_recovery": no_replan_success, "verification": True, "evidence_boundary": False},
            "full_closed_loop": {"capacity_recovery": full_success, "verification": True, "evidence_boundary": True},
        },
        "cases": cases,
    }
    report["acceptance_passed"] = (
        len(cases) >= 24
        and report["failed"] == 0
        and report["metrics"]["top3_candidate_cause_hit_rate"] >= 0.8
        and report["metrics"]["citation_coverage"] >= 0.9
        and report["metrics"]["recovery_gain_percentage_points"] >= 20.0
        and report["metrics"]["simple_regret_improvement"] >= 0.2
    )
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run deterministic scientific closed-loop evals without an online LLM.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="algae_scientific_eval_", ignore_cleanup_errors=True) as tmpdir:
        configure_temp_db(Path(tmpdir) / "scientific_eval.sqlite3")
        report = run_eval(run_seed=args.seed, work_dir=Path(tmpdir))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["acceptance_passed"] else 1)


if __name__ == "__main__":
    main()
