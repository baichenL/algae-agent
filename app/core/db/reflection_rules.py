import sqlite3

from app.core.db.connection import DB_PATH


def insert_reflection_rule(rule_obj: dict) -> int:
    """Insert a reflection rule dict into reflection_rules table. Expects validated dict matching contract."""
    import json
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO reflection_rules (rule, conditions_json, effect_json, confidence, evidence_experiments_json, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rule_obj["rule"],
                json.dumps(rule_obj["conditions"], ensure_ascii=False),
                json.dumps(rule_obj["effect"], ensure_ascii=False),
                float(rule_obj["confidence"]),
                json.dumps(rule_obj.get("evidence_experiments", []), ensure_ascii=False),
                rule_obj.get("notes", ""),
                rule_obj.get("created_at")
            )
        )
        conn.commit()
        return cursor.lastrowid
