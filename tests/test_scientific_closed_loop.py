from __future__ import annotations

import io
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from app.core.db import scientific as scientific_db
from app.core.security import _principal_for_key
from app.services.intent.intent_router import _detect_fragment
from app.services.intent.routing_models import RouteKind
from app.services.scientific.adapters import PredictiveSimulationAdapter
from app.services.scientific.analysis import assess_data_quality, compute_growth_metrics
from app.services.scientific.demo_data import scientific_demo_file
from app.services.scientific.importer import DatasetImportError, parse_scientific_dataset
from app.services.scientific.models import ExperimentCondition, ExperimentDesignSpec
from app.services.scientific.optimizer import fit_response_surface
from app.services.scientific.service import approve_and_simulate_proposal, run_scientific_task


def _dataset(batch_count: int = 6, points: int = 6) -> dict:
    batches = []
    for batch_index in range(batch_count):
        measurements = [
            {
                "elapsed_hours": float(index * 12),
                "metric_name": "biomass",
                "value": float(0.1 * np.exp((0.01 + batch_index * 0.0005) * index * 12)),
                "replicate_index": 1,
                "source_row": batch_index * points + index + 2,
            }
            for index in range(points)
        ]
        batches.append(
            {
                "id": f"b{batch_index}",
                "condition": {"light": float(batch_index % 3), "nitrogen": float(batch_index // 3)},
                "measurements": measurements,
            }
        )
    return {"id": "test", "mapping": {"metric_name": "biomass"}, "batches": batches}


def _import_builtin() -> tuple[dict, list[dict]]:
    content, filename, mapping = scientific_demo_file()
    parsed = parse_scientific_dataset(
        content=content, filename=filename, strain_id="Chlorella_01", mapping=mapping,
    )
    return parsed.dataset, parsed.batches


def test_builtin_xlsx_auto_mapping_is_general_and_builds_25_batches():
    dataset, batches = _import_builtin()
    assert dataset["row_count"] == 275
    assert len(batches) == 25
    assert len(dataset["mapping"]["factor_columns"]) == 4


def test_csv_explicit_mapping_builds_factor_grouped_batches():
    content = b"time,light,biomass\n0,10,0.1\n12,10,0.2\n0,20,0.1\n12,20,0.3\n"
    parsed = parse_scientific_dataset(
        content=content, filename="growth.csv",
        mapping={"time_column": "time", "response_column": "biomass", "factor_columns": ["light"]},
    )
    assert len(parsed.batches) == 2


@pytest.mark.parametrize("filename", ["data.xls", "data.json", "data.exe"])
def test_import_rejects_unsupported_extensions(filename):
    with pytest.raises(DatasetImportError) as exc:
        parse_scientific_dataset(content=b"x", filename=filename)
    assert exc.value.code == "unsupported_extension"


def test_import_returns_mapping_required_instead_of_guessing():
    with pytest.raises(DatasetImportError) as exc:
        parse_scientific_dataset(content=b"a,b\n1,2\n", filename="unknown.csv")
    assert exc.value.code == "mapping_required"


def test_import_rejects_xlsx_formulas():
    import openpyxl
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["time", "biomass"])
    sheet.append([0, "=1+1"])
    buffer = io.BytesIO()
    book.save(buffer)
    with pytest.raises(DatasetImportError) as exc:
        parse_scientific_dataset(content=buffer.getvalue(), filename="formula.xlsx")
    assert exc.value.code == "formula_not_allowed"


@pytest.mark.parametrize(
    "mutation,expected_flag",
    [
        ("short", "insufficient_timepoints"),
        ("duplicate", "duplicate_timepoints"),
        ("nonpositive", "non_positive_growth_value"),
        ("gap", "irregular_sampling_gap"),
        ("reversed", "non_monotonic_source_time"),
    ],
)
def test_quality_fault_injection_is_structured(mutation, expected_flag):
    dataset = _dataset(batch_count=1, points=6)
    values = dataset["batches"][0]["measurements"]
    if mutation == "short":
        del values[3:]
    elif mutation == "duplicate":
        values[1]["elapsed_hours"] = values[0]["elapsed_hours"]
    elif mutation == "nonpositive":
        values[2]["value"] = 0.0
    elif mutation == "gap":
        values[-1]["elapsed_hours"] = 500.0
    elif mutation == "reversed":
        values[1]["elapsed_hours"] = -1.0
    report = assess_data_quality(dataset, "biomass")
    assert expected_flag in {item["code"] for item in report["flags"]}


@pytest.mark.parametrize("growth_rate", [0.005, 0.01, 0.02, 0.03])
def test_growth_metric_uses_positive_log_linear_window(growth_rate):
    dataset = _dataset(batch_count=1, points=8)
    for item in dataset["batches"][0]["measurements"]:
        item["value"] = 0.1 * np.exp(growth_rate * item["elapsed_hours"])
    metric = compute_growth_metrics(dataset, "biomass")[0]
    assert metric["status"] == "ok"
    assert metric["max_specific_growth_rate"] == pytest.approx(growth_rate, rel=1e-5)
    assert metric["fit_r_squared"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "sample_count,expected",
    [(3, "insufficient_design"), (4, "linear"), (6, "quadratic_main_effects"), (10, "full_quadratic_interactions")],
)
def test_response_surface_degrades_by_sample_count(sample_count, expected):
    metrics = [
        {"batch_id": f"b{i}", "condition": {"x": float(i)}, "max_value": float(i * i + 1)}
        for i in range(sample_count)
    ]
    model = fit_response_surface(metrics)
    assert (model.get("mode") or model.get("status")) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "诊断微藻生长异常",
        "为什么这个批次生长变慢",
        "优化下一轮实验",
        "diagnose growth anomaly",
        "experiment optimization for algae",
    ],
)
def test_scientific_phrases_route_to_scientific_task(phrase):
    candidate = _detect_fragment(phrase, None)[0]
    assert candidate.kind == RouteKind.SCIENTIFIC_TASK


def test_full_loop_persists_typed_artifacts_and_repairs_capacity(isolated_sqlite_db):
    dataset, batches = _import_builtin()
    scientific_db.insert_dataset(dataset, batches)
    result = run_scientific_task(dataset_id=dataset["id"], offline_replay=True)
    artifact_types = [item["artifact_type"] for item in result["artifacts"]]
    for required in ("goal_contract", "plan_graph", "observation", "verification_report", "plan_patch", "experiment_design"):
        assert required in artifact_types
    design = [item for item in result["artifacts"] if item["artifact_type"] == "experiment_design"][-1]["payload"]
    assert len(design["conditions"]) * design["replicates"] == 12
    assert result["proposal"]["pending_id"]


def test_approved_plan_runs_offline_replay_and_creates_next_pending(isolated_sqlite_db):
    dataset, batches = _import_builtin()
    scientific_db.insert_dataset(dataset, batches)
    run = run_scientific_task(dataset_id=dataset["id"], offline_replay=True)
    result = approve_and_simulate_proposal(run["proposal"]["pending_id"])
    assert result["status"] == "success"
    assert result["simulation"]["simulation_only"] is True
    assert result["simulation"]["provenance"] == "offline_replay"
    assert result["next_pending"]["pending_id"] != run["proposal"]["pending_id"]


def test_adapter_blocks_capacity_before_simulation():
    design = ExperimentDesignSpec(
        "d", "r", "biomass", "maximize",
        [ExperimentCondition(f"c{i}", {"x": float(i)}, 1.0, 0.1, 0.5, "control" if i == 4 else "candidate") for i in range(5)],
        3, [0, 12], ["prepare_conditions", "inoculate", "incubate", "measure_growth", "record_results"],
    ).freeze()
    report = PredictiveSimulationAdapter().validate_design(design)
    assert report["valid"] is False
    assert report["issues"][0]["code"] == "CAPACITY_EXCEEDED"


def test_auth_test_mode_is_explicit(monkeypatch):
    monkeypatch.setenv("ALGAE_AUTH_MODE", "test")
    assert _principal_for_key(None).role == "approver"


def test_auth_required_mode_rejects_unconfigured_key(monkeypatch):
    monkeypatch.setenv("ALGAE_AUTH_MODE", "required")
    monkeypatch.delenv("ALGAE_API_KEYS_JSON", raising=False)
    assert _principal_for_key(None) is None


def _full_api_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.endpoints import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_required_api_mode_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("ALGAE_AUTH_MODE", "required")
    monkeypatch.setenv("ALGAE_API_KEYS_JSON", json.dumps({"viewer-key": {"name": "v", "role": "viewer"}}))
    assert _full_api_client().get("/api/v1/scientific/datasets").status_code == 401


def test_viewer_cannot_start_run_or_approve(monkeypatch):
    monkeypatch.setenv("ALGAE_AUTH_MODE", "required")
    monkeypatch.setenv("ALGAE_API_KEYS_JSON", json.dumps({"viewer-key": {"name": "v", "role": "viewer"}}))
    headers = {"X-API-Key": "viewer-key"}
    client = _full_api_client()
    assert client.post("/api/v1/scientific/runs", json={"dataset_id": "missing"}, headers=headers).status_code == 403
    assert client.post("/api/v1/strain/confirm", json={"pending_id": 1, "approved": True}, headers=headers).status_code == 403


def test_tampered_pending_design_hash_is_blocked(isolated_sqlite_db):
    from app.core.db import pending_actions

    dataset, batches = _import_builtin()
    scientific_db.insert_dataset(dataset, batches)
    run = run_scientific_task(dataset_id=dataset["id"], offline_replay=True)
    pending_id = run["proposal"]["pending_id"]
    pending = pending_actions.get_pending_action(pending_id)
    payload = pending["payload"]
    payload["data"]["design"]["conditions"][0]["factors"]["tampered"] = 1.0
    with sqlite3.connect(pending_actions.DB_PATH) as conn:
        conn.execute("UPDATE pending_actions SET payload_json = ? WHERE id = ?", (json.dumps(payload), pending_id))
        conn.commit()
    assert approve_and_simulate_proposal(pending_id)["reason"] == "design_hash_mismatch"


def test_mcp_surface_exposes_no_approval_or_execution_tools():
    import asyncio

    from app.mcp.server import mcp
    tools = asyncio.run(mcp.list_tools())
    names = {item.name for item in tools}
    assert {"datasets_list", "growth_diagnosis_start", "experiment_design_preview", "experiment_proposal_request"} <= names
    assert not any("approve" in name or "execute" in name for name in names)
