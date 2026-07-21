from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

from app.core.db.connection import DB_PATH
from app.core.time_utils import local_time_string


def init_control_plane_schema(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_index (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            status TEXT NOT NULL,
            phase TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0,
            cycle INTEGER NOT NULL DEFAULT 0,
            risk_level TEXT NOT NULL DEFAULT 'low',
            simulation_only INTEGER NOT NULL DEFAULT 1,
            source_ref_json TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_index_status_updated ON run_index(status, updated_at)")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            phase TEXT,
            level TEXT NOT NULL DEFAULT 'info',
            payload_json TEXT,
            created_at TEXT,
            UNIQUE(run_id, sequence)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_events_resume ON run_events(run_id, sequence)")


def upsert_run(summary: dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        init_control_plane_schema(conn.cursor())
        conn.execute(
            """
            INSERT INTO run_index
            (id, kind, title, workspace_id, status, phase, progress, cycle, risk_level,
             simulation_only, source_ref_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, workspace_id=excluded.workspace_id, status=excluded.status,
                phase=excluded.phase, progress=excluded.progress, cycle=excluded.cycle,
                risk_level=excluded.risk_level, simulation_only=excluded.simulation_only,
                source_ref_json=excluded.source_ref_json, updated_at=excluded.updated_at
            """,
            (
                summary["id"], summary["kind"], summary["title"], summary.get("workspace_id") or "shared",
                summary["status"], summary["phase"], float(summary.get("progress") or 0),
                int(summary.get("cycle") or 0), summary.get("risk_level") or "low",
                1 if summary.get("simulation_only", True) else 0,
                json.dumps(summary.get("source_ref") or {}, ensure_ascii=False, default=str),
                summary.get("created_at"), summary.get("updated_at"),
            ),
        )


def append_events(run_id: str, events: Iterable[dict[str, Any]]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        init_control_plane_schema(conn.cursor())
        conn.executemany(
            """
            INSERT OR IGNORE INTO run_events
            (run_id, sequence, event_type, phase, level, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id, int(item["sequence"]), item.get("event_type") or "run_event",
                    item.get("phase"), item.get("level") or "info",
                    json.dumps(item.get("payload") or {}, ensure_ascii=False, default=str), item.get("created_at"),
                )
                for item in events
            ],
        )


def append_event(
    run_id: str,
    event_type: str,
    *,
    phase: str | None = None,
    level: str = "info",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one event with a database-assigned monotonic run sequence."""

    created_at = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        init_control_plane_schema(conn.cursor())
        conn.execute("BEGIN IMMEDIATE")
        sequence = int(
            conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO run_events
            (run_id, sequence, event_type, phase, level, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sequence,
                event_type,
                phase,
                level,
                json.dumps(payload or {}, ensure_ascii=False, default=str),
                created_at,
            ),
        )
        conn.commit()
    return {
        "sequence": sequence,
        "event_type": event_type,
        "phase": phase,
        "level": level,
        "payload": payload or {},
        "created_at": created_at,
    }


def list_events(run_id: str, *, after: int = 0) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        init_control_plane_schema(conn.cursor())
        rows = conn.execute(
            "SELECT sequence, event_type, phase, level, payload_json, created_at FROM run_events WHERE run_id = ? AND sequence > ? ORDER BY sequence",
            (run_id, int(after)),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        result.append(item)
    return result
