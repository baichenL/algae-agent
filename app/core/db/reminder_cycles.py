import json
import sqlite3
from typing import Any

from app.core.db.connection import DB_PATH
from app.core.time_utils import display_time_string, local_time_string, parse_time


STATUS_ACTIVE = "active"
STATUS_SILENCED = "silenced"
STATUS_RESOLVED = "resolved"

EVENT_RESERVED = "reserved"
EVENT_SENT = "sent"
EVENT_FAILED = "failed"


def build_cycle_key(strain_id: str, generation_number: int) -> str:
    return f"{strain_id}:generation_{int(generation_number)}"


def ensure_reminder_cycle(
    strain_id: str,
    generation_number: int,
    now: Any = None,
) -> dict:
    cycle_key = build_cycle_key(strain_id, generation_number)
    timestamp = local_time_string(now)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO subculture_reminder_cycles
            (cycle_key, strain_id, generation_number, status, first_due_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cycle_key,
                strain_id,
                int(generation_number),
                STATUS_ACTIVE,
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
        cursor.execute("SELECT * FROM subculture_reminder_cycles WHERE cycle_key = ?", (cycle_key,))
        return _cycle_row_to_dict(cursor.fetchone())


def get_reminder_cycle(cycle_key: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM subculture_reminder_cycles WHERE cycle_key = ?", (cycle_key,))
        row = cursor.fetchone()
        return _cycle_row_to_dict(row) if row else None


def list_reminder_cycles_for_strains(strains: list[dict]) -> dict[str, dict]:
    keys = [
        build_cycle_key(item.get("strain_id"), int(item.get("generation_number") or 0))
        for item in strains
        if item.get("strain_id") and item.get("generation_number") is not None
    ]
    if not keys:
        return {}

    placeholders = ",".join("?" for _ in keys)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT * FROM subculture_reminder_cycles WHERE cycle_key IN ({placeholders})",
            keys,
        )
        return {
            row["cycle_key"]: _cycle_row_to_dict(row)
            for row in cursor.fetchall()
        }


def reserve_reminder_window(
    strain_id: str,
    generation_number: int,
    window_key: str,
    sent_date: str,
    source: str,
    now: Any = None,
    daily_limit: int = 3,
    cycle_limit: int = 6,
    max_reminder_days: int = 2,
    metadata: dict | None = None,
) -> tuple[bool, str, dict | None]:
    cycle = ensure_reminder_cycle(strain_id, generation_number, now)
    if not cycle:
        return False, "cycle_not_found", None
    if cycle["status"] != STATUS_ACTIVE:
        return False, f"cycle_{cycle['status']}", cycle

    if int(cycle.get("total_sent_count") or 0) >= cycle_limit:
        silence_reminder_cycle(cycle["cycle_key"], "cycle_email_cap", now)
        return False, "cycle_email_cap", get_reminder_cycle(cycle["cycle_key"])

    if _reminder_days_exceeded(cycle, sent_date, max_reminder_days):
        silence_reminder_cycle(cycle["cycle_key"], "cycle_day_cap", now)
        return False, "cycle_day_cap", get_reminder_cycle(cycle["cycle_key"])

    daily_count = get_cycle_sent_count_for_date(cycle["cycle_key"], sent_date)
    if daily_count >= daily_limit:
        return False, "daily_email_cap", cycle

    timestamp = local_time_string(now)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO subculture_reminder_events
                (cycle_key, window_key, sent_date, event_type, source, reason, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle["cycle_key"],
                    window_key,
                    sent_date,
                    EVENT_RESERVED,
                    source,
                    "eligible",
                    timestamp,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "duplicate_window", cycle

    return True, "reserved", cycle


def mark_reserved_events_sent(
    cycle_keys: list[str],
    window_key: str,
    email_log_id: str | None,
    now: Any = None,
    cycle_limit: int = 6,
) -> None:
    if not cycle_keys:
        return
    timestamp = local_time_string(now)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for cycle_key in cycle_keys:
            cursor.execute(
                """
                UPDATE subculture_reminder_events
                SET event_type = ?, email_log_id = ?, reason = ?
                WHERE cycle_key = ? AND window_key = ? AND event_type = ?
                """,
                (EVENT_SENT, email_log_id, "sent", cycle_key, window_key, EVENT_RESERVED),
            )
            cursor.execute(
                """
                UPDATE subculture_reminder_cycles
                SET
                    first_sent_at = COALESCE(first_sent_at, ?),
                    last_sent_at = ?,
                    total_sent_count = total_sent_count + 1,
                    updated_at = ?
                WHERE cycle_key = ?
                """,
                (timestamp, timestamp, timestamp, cycle_key),
            )
            cursor.execute(
                """
                UPDATE subculture_reminder_cycles
                SET status = ?, silenced_at = ?, silenced_reason = ?, updated_at = ?
                WHERE cycle_key = ? AND total_sent_count >= ? AND status = ?
                """,
                (
                    STATUS_SILENCED,
                    timestamp,
                    "cycle_email_cap",
                    timestamp,
                    cycle_key,
                    cycle_limit,
                    STATUS_ACTIVE,
                ),
            )
        conn.commit()


def mark_reserved_events_failed(
    cycle_keys: list[str],
    window_key: str,
    reason: str,
    now: Any = None,
) -> None:
    if not cycle_keys:
        return
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for cycle_key in cycle_keys:
            cursor.execute(
                """
                UPDATE subculture_reminder_events
                SET event_type = ?, reason = ?
                WHERE cycle_key = ? AND window_key = ? AND event_type = ?
                """,
                (EVENT_FAILED, reason, cycle_key, window_key, EVENT_RESERVED),
            )
        conn.commit()


def silence_reminder_cycle(cycle_key: str, reason: str, now: Any = None) -> bool:
    timestamp = local_time_string(now)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE subculture_reminder_cycles
            SET status = ?, silenced_at = ?, silenced_reason = ?, updated_at = ?
            WHERE cycle_key = ? AND status = ?
            """,
            (STATUS_SILENCED, timestamp, reason, timestamp, cycle_key, STATUS_ACTIVE),
        )
        conn.commit()
        return cursor.rowcount > 0


def resolve_reminder_cycles_before_generation(
    strain_id: str,
    generation_number: int,
    now: Any = None,
) -> int:
    timestamp = local_time_string(now)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE subculture_reminder_cycles
            SET status = ?, resolved_at = ?, updated_at = ?
            WHERE strain_id = ?
              AND generation_number < ?
              AND status IN (?, ?)
            """,
            (
                STATUS_RESOLVED,
                timestamp,
                timestamp,
                strain_id,
                int(generation_number),
                STATUS_ACTIVE,
                STATUS_SILENCED,
            ),
        )
        conn.commit()
        return cursor.rowcount


def resolve_reminder_cycle(
    strain_id: str,
    generation_number: int,
    now: Any = None,
) -> bool:
    timestamp = local_time_string(now)
    cycle_key = build_cycle_key(strain_id, generation_number)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE subculture_reminder_cycles
            SET status = ?, resolved_at = ?, updated_at = ?
            WHERE cycle_key = ? AND status IN (?, ?)
            """,
            (
                STATUS_RESOLVED,
                timestamp,
                timestamp,
                cycle_key,
                STATUS_ACTIVE,
                STATUS_SILENCED,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_cycle_sent_count_for_date(cycle_key: str, sent_date: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM subculture_reminder_events
            WHERE cycle_key = ? AND sent_date = ? AND event_type = ?
            """,
            (cycle_key, sent_date, EVENT_SENT),
        )
        return int(cursor.fetchone()[0] or 0)


def _cycle_row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if not row:
        return None
    cycle = dict(row)
    for key in [
        "first_due_at",
        "first_sent_at",
        "last_sent_at",
        "silenced_at",
        "resolved_at",
        "created_at",
        "updated_at",
    ]:
        cycle[key] = display_time_string(cycle.get(key))
    return cycle


def _reminder_days_exceeded(cycle: dict, sent_date: str, max_reminder_days: int) -> bool:
    if max_reminder_days <= 0:
        return False
    first_sent_at = parse_time(cycle.get("first_sent_at"))
    if not first_sent_at:
        return False
    try:
        current_date = parse_time(sent_date).date()
    except AttributeError:
        return False
    return (current_date - first_sent_at.date()).days >= max_reminder_days
