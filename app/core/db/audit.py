import sqlite3

from app.core.db.connection import DB_PATH
from app.core.time_utils import display_time_string


def get_last_db_operation() -> dict:
    """Return the most recent audit record for algae_status operations, or None if none."""
    import json
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM algae_audit ORDER BY performed_at DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return None
        r = dict(row)
        r["performed_at"] = display_time_string(r.get("performed_at"))
        try:
            r["details"] = json.loads(r.pop("details_json") or "{}")
        except Exception:
            r["details"] = {}
        return r
