import sqlite3

from app.core.db.connection import DB_PATH


def insert_experiment(record: dict) -> int:
    """Insert an experiment record into experiments table. Returns inserted id."""
    import json
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO experiments (strain, generation, status, media_components_json, temperature, light_intensity, od_readings_json, hardware_logs, created_at, reflected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("strain"),
                record.get("generation"),
                record.get("status"),
                json.dumps(record.get("media_components", {}), ensure_ascii=False),
                record.get("temperature"),
                record.get("light_intensity"),
                json.dumps(record.get("od_readings", []), ensure_ascii=False),
                json.dumps(record.get("hardware_logs", []), ensure_ascii=False),
                record.get("created_at"),
                0
            )
        )
        conn.commit()
        return cursor.lastrowid

def get_experiment_by_id(experiment_id: int) -> dict:
    import json
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        # decode json fields
        try:
            result["media_components"] = json.loads(result.pop("media_components_json") or "{}")
        except Exception:
            result["media_components"] = {}
        try:
            result["od_readings"] = json.loads(result.pop("od_readings_json") or "[]")
        except Exception:
            result["od_readings"] = []
        try:
            result["hardware_logs"] = json.loads(result.pop("hardware_logs") or "[]")
        except Exception:
            result["hardware_logs"] = []
        return result

def get_experiments_by_strain(strain: str) -> list:
    import json
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE strain = ? ORDER BY created_at DESC", (strain,))
        rows = cursor.fetchall()
        out = []
        for row in rows:
            r = dict(row)
            try:
                r["media_components"] = json.loads(r.pop("media_components_json") or "{}")
            except Exception:
                r["media_components"] = {}
            try:
                r["od_readings"] = json.loads(r.pop("od_readings_json") or "[]")
            except Exception:
                r["od_readings"] = []
            try:
                r["hardware_logs"] = json.loads(r.pop("hardware_logs") or "[]")
            except Exception:
                r["hardware_logs"] = []
            out.append(r)
        return out


def get_recent_experiments(strain: str = None, limit: int = 5) -> list:
    import json
    safe_limit = max(0, min(int(limit or 5), 50))
    if safe_limit == 0:
        return []

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if strain:
            cursor.execute(
                "SELECT * FROM experiments WHERE strain = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (strain, safe_limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC, id DESC LIMIT ?",
                (safe_limit,),
            )
        rows = cursor.fetchall()
        out = []
        for row in rows:
            r = dict(row)
            try:
                r["media_components"] = json.loads(r.pop("media_components_json") or "{}")
            except Exception:
                r["media_components"] = {}
            try:
                r["od_readings"] = json.loads(r.pop("od_readings_json") or "[]")
            except Exception:
                r["od_readings"] = []
            try:
                hardware_logs = json.loads(r.pop("hardware_logs") or "[]")
            except Exception:
                hardware_logs = []
            r["hardware_log_count"] = len(hardware_logs) if isinstance(hardware_logs, list) else 0
            out.append(r)
        return out

