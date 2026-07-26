from __future__ import annotations

import json
import uuid
from typing import Any

from app.core.db.connection import connect
from app.core.effects import stable_hash
from app.core.time_utils import local_time_string


def save_agent_artifact(
    *,
    agent_run_id: str | None,
    artifact_type: str,
    payload: dict[str, Any],
    status: str = "active",
    artifact_id: str | None = None,
) -> dict[str, Any]:
    artifact_id = artifact_id or str(uuid.uuid4())
    now = local_time_string()
    with connect(row_factory=True) as conn:
        previous = conn.execute(
            """
            SELECT COALESCE(MAX(version), 0) AS version
            FROM agent_artifacts_v2
            WHERE agent_run_id IS ? AND artifact_type = ?
            """,
            (agent_run_id, artifact_type),
        ).fetchone()
        version = int(previous["version"] or 0) + 1
        conn.execute(
            """
            INSERT INTO agent_artifacts_v2 (
                artifact_id, agent_run_id, artifact_type, version, status,
                payload_json, payload_hash, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                agent_run_id,
                artifact_type,
                version,
                status,
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                stable_hash(payload),
                now,
                now,
            ),
        )
        conn.commit()
    return {
        "artifact_id": artifact_id,
        "agent_run_id": agent_run_id,
        "artifact_type": artifact_type,
        "version": version,
        "status": status,
        "payload_hash": stable_hash(payload),
        "created_at": now,
    }


def get_agent_artifact(artifact_id: str) -> dict[str, Any] | None:
    with connect(row_factory=True) as conn:
        row = conn.execute(
            "SELECT * FROM agent_artifacts_v2 WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["payload"] = json.loads(result.pop("payload_json") or "{}")
    return result


def list_agent_artifacts(
    *,
    agent_run_id: str,
    artifact_type: str | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM agent_artifacts_v2 WHERE agent_run_id = ?"
    params: list[Any] = [agent_run_id]
    if artifact_type:
        sql += " AND artifact_type = ?"
        params.append(artifact_type)
    sql += " ORDER BY created_at, version"
    with connect(row_factory=True) as conn:
        rows = conn.execute(sql, params).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        results.append(item)
    return results
