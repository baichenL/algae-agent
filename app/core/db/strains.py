# app/core/db/strains.py
# 璐熻矗涓庡搧绯荤浉鍏崇殑鏁版嵁搴撴搷浣滐紝鏌ヨ鍝佺郴鐘舵€併€佸垪鍑烘墍鏈夊搧绯汇€佹洿鏂颁紶浠ｇ姸鎬併€佹柊澧炴垨鍒犻櫎鍝佺郴
import datetime
import json
import sqlite3

from app.core.workspaces import DOMAIN_DATABASE_PATH as DB_PATH
from app.core.db.domain_connection import connect_domain_readonly, connect_domain_writer
from app.core.db.reminder_cycles import (
    resolve_reminder_cycle,
    resolve_reminder_cycles_before_generation,
)
from app.core.time_utils import days_since_time, local_time_string, time_days_ago
from app.services.observability.error_events import record_error_event


def _with_dynamic_days(row: dict) -> dict:
    if not row:
        return row
    fallback_days = int(row.get("days_since_last_subculture") or 0)
    row["days_since_last_subculture"] = days_since_time(
        row.get("last_subculture_time"),
        fallback=fallback_days,
    )
    return row


def get_algae_status(strain_id: str) -> dict | None:
    """
    銆愪弗鏍兼煡璇㈡ā寮忋€?
    杩斿洖鍖呭惈涓嫳鏂囧悕绉板湪鍐呯殑瀹屾暣瀛楀吀銆傚鏋滄煡涓嶅埌锛岀洿鎺ョ啍鏂繑鍥?None銆?
    """
    with connect_domain_readonly(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM algae_status WHERE strain_id = ?", (strain_id,))
        row = cursor.fetchone()
        return _with_dynamic_days(dict(row)) if row else None

def list_algae_status() -> list:
    """
    Return all rows from algae_status as list of dicts.
    """
    with connect_domain_readonly(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM algae_status ORDER BY strain_id")
        rows = cursor.fetchall()
        return [_with_dynamic_days(dict(r)) for r in rows]

def update_algae_status(strain_id: str, generation_number: int, days_since_last_subculture: int, last_time: str):
    """
    銆愪弗鏍兼洿鏂版ā寮忋€?
    浠呮洿鏂板姩鎬佸疄楠屾暟鎹紙浠ｆ暟銆侀棿闅斿ぉ鏁般€佹渶鍚庝紶浠ｆ椂闂达級锛屼笉淇敼鏍稿績鐨勪腑鑻辨枃鍏冩暟鎹€?
    """
    previous_generation = None
    with connect_domain_writer(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT generation_number FROM algae_status WHERE strain_id = ?", (strain_id,))
        previous_row = cursor.fetchone()
        if previous_row:
            previous_generation = int(previous_row[0])
        cursor.execute("""
            UPDATE algae_status 
            SET generation_number = ?, days_since_last_subculture = ?, last_subculture_time = ?
            WHERE strain_id = ?
        """, (generation_number, days_since_last_subculture, last_time, strain_id))
        conn.commit()
        # 鍐欏叆瀹¤璁板綍
        try:
            import json as _json, datetime as _dt
            cursor.execute(
                "INSERT INTO algae_audit (strain_id, action, details_json, performed_at) VALUES (?, ?, ?, ?)",
                (strain_id, 'update_status', _json.dumps({'generation_number': generation_number, 'days_since_last_subculture': days_since_last_subculture, 'last_subculture_time': last_time}, ensure_ascii=False), local_time_string())
            )
            conn.commit()
        except Exception as exc:
            record_error_event(
                layer="memory_db",
                component="db.strains",
                operation="audit_update_status",
                severity="warning",
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={"strain_id": strain_id},
            )
            pass

    if previous_generation is not None:
        if int(generation_number) > previous_generation:
            resolve_reminder_cycles_before_generation(strain_id, int(generation_number))
        elif int(days_since_last_subculture or 0) == 0:
            resolve_reminder_cycle(strain_id, int(generation_number))

def delete_algae_strain(strain_id: str) -> bool:
    with connect_domain_writer(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM algae_status WHERE strain_id = ?", (strain_id,))
        affected = cursor.rowcount
        conn.commit()
        # 鍐欏叆瀹¤璁板綍锛堝鏋滅‘瀹炲垹闄や簡锛?
        try:
            if affected and affected > 0:
                import json as _json, datetime as _dt
                cursor.execute(
                    "INSERT INTO algae_audit (strain_id, action, details_json, performed_at) VALUES (?, ?, ?, ?)",
                    (strain_id, 'delete', _json.dumps({'deleted': True}, ensure_ascii=False), local_time_string())
                )
                conn.commit()
        except Exception as exc:
            record_error_event(
                layer="memory_db",
                component="db.strains",
                operation="audit_delete_strain",
                severity="warning",
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={"strain_id": strain_id, "affected": affected},
            )
            pass
        return affected > 0

def add_algae_strain(strain_id: str, name_cn: str, name_en: str, generation_number: int = 1, days_since_last_subculture: int = 0, last_subculture_time: str = None) -> bool:
    """Add a new algae strain to algae_status. Returns True if inserted or updated."""
    if last_subculture_time is None:
        last_subculture_time = time_days_ago(days_since_last_subculture)

    # 鍒ゆ柇鏄柊澧炶繕鏄洿鏂帮紙鐢ㄤ簬瀹¤璁板綍锛?
    existed = get_algae_status(strain_id)
    action_type = 'update' if existed else 'create'

    with connect_domain_writer(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO algae_status (strain_id, name_cn, name_en, generation_number, days_since_last_subculture, last_subculture_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (strain_id, name_cn, name_en, generation_number, days_since_last_subculture, last_subculture_time)
        )
        conn.commit()
        # 鍐欏叆瀹¤璁板綍
        try:
            import json as _json, datetime as _dt
            cursor.execute(
                "INSERT INTO algae_audit (strain_id, action, details_json, performed_at) VALUES (?, ?, ?, ?)",
                (strain_id, action_type, _json.dumps({'name_cn': name_cn, 'name_en': name_en, 'generation_number': generation_number, 'days_since_last_subculture': days_since_last_subculture}, ensure_ascii=False), local_time_string())
            )
            conn.commit()
        except Exception as exc:
            record_error_event(
                layer="memory_db",
                component="db.strains",
                operation="audit_add_algae_strain",
                severity="warning",
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={"strain_id": strain_id, "action_type": action_type},
            )
            pass
        return True


def apply_strain_mutation_once(
    *,
    action_type: str,
    data: dict,
    execution_idempotency_key: str,
) -> dict:
    """Apply one approved strain mutation and its execution marker atomically."""
    if action_type not in {"add_strain", "update_strain", "delete_strain"}:
        raise ValueError("unsupported_strain_mutation")
    with connect_domain_writer(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        prior = conn.execute(
            """
            SELECT result_json FROM domain_executions
            WHERE execution_idempotency_key = ?
            """,
            (execution_idempotency_key,),
        ).fetchone()
        if prior:
            conn.commit()
            result = json.loads(prior["result_json"])
            return {**result, "idempotent": True}

        strain_id = str(data.get("strain_id") or "")
        if not strain_id:
            conn.rollback()
            return {"status": "error", "reason": "strain_id_required"}
        existing_row = conn.execute(
            "SELECT * FROM algae_status WHERE strain_id = ?",
            (strain_id,),
        ).fetchone()
        existing = dict(existing_row) if existing_row else None

        if action_type == "add_strain":
            last_time = data.get("last_subculture_time") or time_days_ago(
                int(data.get("days_since_last_subculture") or 0)
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO algae_status (
                    strain_id, name_cn, name_en, generation_number,
                    days_since_last_subculture, last_subculture_time
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    strain_id,
                    data.get("name_cn"),
                    data.get("name_en"),
                    int(data.get("generation_number") or 1),
                    int(data.get("days_since_last_subculture") or 0),
                    last_time,
                ),
            )
            audit_action = "update" if existing else "create"
            result_strain_id = strain_id
        elif action_type == "update_strain":
            if not existing:
                conn.rollback()
                return {"status": "error", "reason": "not_found"}
            result_strain_id = str(data.get("new_strain_id") or strain_id)
            days = int(
                data.get("days_since_last_subculture")
                if data.get("days_since_last_subculture") is not None
                else existing.get("days_since_last_subculture") or 0
            )
            last_time = (
                data.get("last_subculture_time")
                or (
                    time_days_ago(days)
                    if data.get("days_since_last_subculture") is not None
                    else existing.get("last_subculture_time")
                )
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO algae_status (
                    strain_id, name_cn, name_en, generation_number,
                    days_since_last_subculture, last_subculture_time
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result_strain_id,
                    data.get("name_cn") or existing.get("name_cn"),
                    data.get("name_en") or existing.get("name_en"),
                    int(data.get("generation_number") or existing.get("generation_number") or 1),
                    days,
                    last_time,
                ),
            )
            if result_strain_id != strain_id:
                conn.execute(
                    "DELETE FROM algae_status WHERE strain_id = ?",
                    (strain_id,),
                )
            audit_action = "update"
        else:
            if not existing:
                conn.rollback()
                return {"status": "error", "reason": "not_found"}
            conn.execute(
                "DELETE FROM algae_status WHERE strain_id = ?",
                (strain_id,),
            )
            result_strain_id = strain_id
            audit_action = "delete"

        conn.execute(
            """
            INSERT INTO algae_audit (
                strain_id, action, details_json, performed_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                strain_id,
                audit_action,
                json.dumps(
                    {
                        "execution_idempotency_key": execution_idempotency_key,
                        "action_type": action_type,
                        "data": data,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                local_time_string(),
            ),
        )
        result = {
            "status": "success",
            "action": "approved",
            "executed": True,
            "action_type": action_type,
            "strain_id": result_strain_id,
            "execution_idempotency_key": execution_idempotency_key,
        }
        conn.execute(
            """
            INSERT INTO domain_executions (
                execution_idempotency_key, action_type, result_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                execution_idempotency_key,
                action_type,
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                local_time_string(),
            ),
        )
        conn.commit()
        return result

