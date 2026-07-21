from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable

from app.core.db.connection import DB_PATH
from app.core.time_utils import local_time_string


def init_scientific_schema(cursor: sqlite3.Cursor) -> None:
    """Create the additive schema used by the scientific closed loop."""
    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS scientific_datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            strain_id TEXT,
            source_file TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            mapping_json TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'ready',
            created_at TEXT NOT NULL,
            UNIQUE(content_hash, strain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_scientific_datasets_created
        ON scientific_datasets(created_at, id);

        CREATE TABLE IF NOT EXISTS culture_batches (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            strain_id TEXT,
            condition_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'observed',
            provenance TEXT NOT NULL DEFAULT 'real_import',
            created_at TEXT NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES scientific_datasets(id)
        );
        CREATE INDEX IF NOT EXISTS idx_culture_batches_dataset
        ON culture_batches(dataset_id, id);

        CREATE TABLE IF NOT EXISTS culture_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            elapsed_hours REAL NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            replicate_index INTEGER NOT NULL DEFAULT 1,
            quality_flag TEXT,
            source_row INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES culture_batches(id),
            UNIQUE(batch_id, elapsed_hours, metric_name, replicate_index)
        );
        CREATE INDEX IF NOT EXISTS idx_culture_measurements_batch_time
        ON culture_measurements(batch_id, metric_name, elapsed_hours);

        CREATE TABLE IF NOT EXISTS scientific_runs (
            id TEXT PRIMARY KEY,
            agent_run_id TEXT,
            session_id TEXT,
            dataset_id TEXT NOT NULL,
            status TEXT NOT NULL,
            mode TEXT NOT NULL,
            cycle_index INTEGER NOT NULL DEFAULT 0,
            goal_json TEXT NOT NULL,
            adapter_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES scientific_datasets(id)
        );
        CREATE INDEX IF NOT EXISTS idx_scientific_runs_dataset
        ON scientific_runs(dataset_id, created_at);

        CREATE TABLE IF NOT EXISTS scientific_artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scientific_run_id TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            version INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(scientific_run_id) REFERENCES scientific_runs(id),
            UNIQUE(scientific_run_id, artifact_type, version)
        );
        CREATE INDEX IF NOT EXISTS idx_scientific_artifacts_run
        ON scientific_artifacts(scientific_run_id, id);

        CREATE TABLE IF NOT EXISTS lab_devices (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            adapter_type TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            capabilities_json TEXT NOT NULL,
            constraints_json TEXT NOT NULL,
            last_seen_at TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS virtual_experiment_results (
            id TEXT PRIMARY KEY,
            scientific_run_id TEXT NOT NULL,
            pending_id INTEGER,
            design_hash TEXT NOT NULL,
            adapter_id TEXT NOT NULL,
            provenance TEXT NOT NULL,
            simulation_only INTEGER NOT NULL DEFAULT 1,
            measurements_json TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(scientific_run_id) REFERENCES scientific_runs(id),
            UNIQUE(scientific_run_id, design_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_virtual_results_run
        ON virtual_experiment_results(scientific_run_id, created_at);
        """
    )
    now = local_time_string()
    cursor.execute(
        """
        INSERT OR IGNORE INTO lab_devices
        (id, name, adapter_type, mode, status, capabilities_json, constraints_json, last_seen_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sim-algae-lab-v1",
            "Simulated algae cultivation lab",
            "predictive_simulation",
            "simulation_only",
            "online",
            json.dumps(
                [
                    "prepare_conditions",
                    "inoculate",
                    "incubate",
                    "measure_growth",
                    "record_results",
                ],
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "max_culture_vessels": 12,
                    "supported_metrics": ["biomass", "od600", "od750"],
                },
                ensure_ascii=False,
            ),
            now,
            now,
        ),
    )


def _loads(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return default


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    for key in (
        "mapping_json",
        "condition_json",
        "goal_json",
        "payload_json",
        "capabilities_json",
        "constraints_json",
        "measurements_json",
        "summary_json",
    ):
        if key in item:
            item[key.removesuffix("_json")] = _loads(item.pop(key), {} if key != "capabilities_json" else [])
    if "simulation_only" in item:
        item["simulation_only"] = bool(item["simulation_only"])
    return item


def insert_dataset(dataset: dict[str, Any], batches: Iterable[dict[str, Any]]) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        now = local_time_string()
        conn.execute(
            """
            INSERT INTO scientific_datasets
            (id, name, strain_id, source_file, content_hash, mapping_json, row_count, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset["id"], dataset["name"], dataset.get("strain_id"), dataset["source_file"],
                dataset["content_hash"], json.dumps(dataset["mapping"], ensure_ascii=False, sort_keys=True),
                int(dataset.get("row_count") or 0), dataset.get("status", "ready"), now,
            ),
        )
        for batch in batches:
            conn.execute(
                """
                INSERT INTO culture_batches
                (id, dataset_id, strain_id, condition_json, status, provenance, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch["id"], dataset["id"], batch.get("strain_id") or dataset.get("strain_id"),
                    json.dumps(batch.get("condition") or {}, ensure_ascii=False, sort_keys=True),
                    batch.get("status", "observed"), batch.get("provenance", "real_import"), now,
                ),
            )
            for measurement in batch.get("measurements") or []:
                conn.execute(
                    """
                    INSERT INTO culture_measurements
                    (batch_id, elapsed_hours, metric_name, value, unit, replicate_index, quality_flag, source_row, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch["id"], float(measurement["elapsed_hours"]), measurement["metric_name"],
                        float(measurement["value"]), measurement.get("unit"),
                        int(measurement.get("replicate_index") or 1), measurement.get("quality_flag"),
                        measurement.get("source_row"), now,
                    ),
                )
        conn.commit()
    return str(dataset["id"])


def list_datasets() -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT d.*, COUNT(DISTINCT b.id) AS batch_count,
                   COUNT(m.id) AS measurement_count
            FROM scientific_datasets d
            LEFT JOIN culture_batches b ON b.dataset_id = d.id
            LEFT JOIN culture_measurements m ON m.batch_id = b.id
            GROUP BY d.id ORDER BY d.created_at DESC, d.id DESC
            """
        ).fetchall()
        return [_row(row) for row in rows]


def get_dataset(dataset_id: str, *, include_measurements: bool = True) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        dataset = _row(conn.execute("SELECT * FROM scientific_datasets WHERE id = ?", (dataset_id,)).fetchone())
        if not dataset:
            return None
        batch_rows = conn.execute(
            "SELECT * FROM culture_batches WHERE dataset_id = ? ORDER BY id", (dataset_id,)
        ).fetchall()
        batches = []
        for raw in batch_rows:
            batch = _row(raw)
            if include_measurements:
                measurements = conn.execute(
                    """
                    SELECT elapsed_hours, metric_name, value, unit, replicate_index, quality_flag, source_row
                    FROM culture_measurements WHERE batch_id = ?
                    ORDER BY elapsed_hours, replicate_index, id
                    """,
                    (batch["id"],),
                ).fetchall()
                batch["measurements"] = [dict(item) for item in measurements]
            batches.append(batch)
        dataset["batches"] = batches
        return dataset


def insert_scientific_run(run: dict[str, Any]) -> str:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO scientific_runs
            (id, agent_run_id, session_id, dataset_id, status, mode, cycle_index, goal_json, adapter_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run["id"], run.get("agent_run_id"), run.get("session_id"), run["dataset_id"],
                run.get("status", "running"), run.get("mode", "diagnose_and_optimize"),
                int(run.get("cycle_index") or 0), json.dumps(run["goal"], ensure_ascii=False, sort_keys=True),
                run.get("adapter_id", "sim-algae-lab-v1"), now, now,
            ),
        )
        conn.commit()
    return str(run["id"])


def update_scientific_run(run_id: str, *, status: str | None = None, cycle_index: int | None = None) -> None:
    updates = ["updated_at = ?"]
    values: list[Any] = [local_time_string()]
    if status is not None:
        updates.append("status = ?")
        values.append(status)
    if cycle_index is not None:
        updates.append("cycle_index = ?")
        values.append(int(cycle_index))
    values.append(run_id)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"UPDATE scientific_runs SET {', '.join(updates)} WHERE id = ?", values)
        conn.commit()


def add_artifact(run_id: str, artifact_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with sqlite3.connect(DB_PATH) as conn:
        version = int(conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM scientific_artifacts WHERE scientific_run_id = ? AND artifact_type = ?",
            (run_id, artifact_type),
        ).fetchone()[0])
        conn.execute(
            """
            INSERT INTO scientific_artifacts
            (scientific_run_id, artifact_type, version, payload_json, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, artifact_type, version, canonical, content_hash, local_time_string()),
        )
        conn.commit()
    return {"artifact_type": artifact_type, "version": version, "content_hash": content_hash, "payload": payload}


def get_scientific_run(run_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        run = _row(conn.execute("SELECT * FROM scientific_runs WHERE id = ?", (run_id,)).fetchone())
        if not run:
            return None
        artifacts = conn.execute(
            "SELECT * FROM scientific_artifacts WHERE scientific_run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        run["artifacts"] = [_row(item) for item in artifacts]
        results = conn.execute(
            "SELECT * FROM virtual_experiment_results WHERE scientific_run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()
        run["virtual_results"] = [_row(item) for item in results]
        return run


def list_scientific_runs(*, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit or 100), 500))
    clause = "WHERE status = ?" if status else ""
    params: tuple[Any, ...] = (status, limit) if status else (limit,)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM scientific_runs {clause} ORDER BY created_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
        items = []
        for row in rows:
            item = _row(row)
            items.append(item)
        return items


def list_lab_devices() -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [_row(row) for row in conn.execute("SELECT * FROM lab_devices ORDER BY id").fetchall()]


def insert_virtual_result(result: dict[str, Any]) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO virtual_experiment_results
            (id, scientific_run_id, pending_id, design_hash, adapter_id, provenance,
             simulation_only, measurements_json, summary_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                result["id"], result["scientific_run_id"], result.get("pending_id"), result["design_hash"],
                result["adapter_id"], result["provenance"],
                json.dumps(result.get("measurements") or [], ensure_ascii=False, sort_keys=True),
                json.dumps(result.get("summary") or {}, ensure_ascii=False, sort_keys=True),
                local_time_string(),
            ),
        )
        conn.commit()
    return str(result["id"])
