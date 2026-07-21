from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from app.core.db import connection
from app.core.time_utils import local_time_string


def _decode_message(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    try:
        item["structured"] = json.loads(item.pop("structured_json") or "null")
    except Exception:
        item["structured"] = None
    return item


def create_conversation(owner: str, *, workspace_id: str = "shared", title: str = "新对话", legacy_session_id: str | None = None) -> dict:
    conversation_id = str(uuid.uuid4())
    now = local_time_string()
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            INSERT INTO assistant_conversations
            (id, owner, workspace_id, title, status, legacy_session_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (conversation_id, owner, workspace_id, title or "新对话", legacy_session_id, now, now),
        )
        row = conn.execute("SELECT * FROM assistant_conversations WHERE id = ?", (conversation_id,)).fetchone()
        conn.commit()
        return dict(row)


def get_conversation(conversation_id: str) -> dict | None:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM assistant_conversations WHERE id = ?", (conversation_id,)).fetchone()
        return dict(row) if row else None


def get_legacy_conversation(legacy_session_id: str) -> dict | None:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM assistant_conversations WHERE legacy_session_id = ?", (legacy_session_id,)
        ).fetchone()
        return dict(row) if row else None


def list_conversations(owner: str, *, workspace_id: str = "shared", status: str = "active") -> list[dict]:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.*, COUNT(m.id) AS message_count
            FROM assistant_conversations c
            LEFT JOIN assistant_messages m ON m.conversation_id = c.id
            WHERE c.owner = ? AND c.workspace_id = ? AND c.status = ?
            GROUP BY c.id
            ORDER BY COALESCE(c.last_message_at, c.updated_at) DESC
            """,
            (owner, workspace_id, status),
        ).fetchall()
        return [dict(row) for row in rows]


def list_messages(conversation_id: str, *, limit: int | None = None) -> list[dict]:
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if limit:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM assistant_messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (conversation_id, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM assistant_messages WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        return [_decode_message(row) for row in rows]


def append_message(
    conversation_id: str,
    *,
    role: str,
    content: str,
    run_id: str | None = None,
    structured: Any = None,
    client_message_id: str | None = None,
    status: str = "completed",
    operation_id: str | None = None,
    task_id: str | None = None,
) -> dict:
    now = local_time_string()
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            INSERT OR IGNORE INTO assistant_messages
            (conversation_id, role, content, run_id, structured_json, client_message_id, created_at,
             status, operation_id, task_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                run_id,
                json.dumps(structured, ensure_ascii=False, default=str) if structured is not None else None,
                client_message_id,
                now,
                status,
                operation_id,
                task_id,
            ),
        )
        if role == "user":
            current = conn.execute(
                "SELECT title FROM assistant_conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            title = str(content).strip().replace("\n", " ")[:40] or "新对话"
            if current and current["title"] == "新对话":
                conn.execute("UPDATE assistant_conversations SET title = ? WHERE id = ?", (title, conversation_id))
        conn.execute(
            "UPDATE assistant_conversations SET updated_at = ?, last_message_at = ? WHERE id = ?",
            (now, now, conversation_id),
        )
        row = conn.execute(
            "SELECT * FROM assistant_messages WHERE conversation_id = ? AND client_message_id IS ? ORDER BY id DESC LIMIT 1",
            (conversation_id, client_message_id),
        ).fetchone()
        conn.commit()
        return _decode_message(row)


def update_message(
    message_id: int,
    *,
    content: str | None = None,
    run_id: str | None = None,
    structured: Any = None,
    status: str | None = None,
    operation_id: str | None = None,
    task_id: str | None = None,
) -> dict | None:
    updates: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("content", content),
        ("run_id", run_id),
        ("status", status),
        ("operation_id", operation_id),
        ("task_id", task_id),
    ):
        if value is not None:
            updates.append(f"{column} = ?")
            params.append(value)
    if structured is not None:
        updates.append("structured_json = ?")
        params.append(json.dumps(structured, ensure_ascii=False, default=str))
    if not updates:
        return None
    params.append(message_id)
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(f"UPDATE assistant_messages SET {', '.join(updates)} WHERE id = ?", params)
        row = conn.execute("SELECT * FROM assistant_messages WHERE id = ?", (message_id,)).fetchone()
        if row:
            conn.execute(
                "UPDATE assistant_conversations SET updated_at = ?, last_message_at = ? WHERE id = ?",
                (local_time_string(), local_time_string(), row["conversation_id"]),
            )
        conn.commit()
        return _decode_message(row) if row else None


def update_conversation(conversation_id: str, *, title: str | None = None, status: str | None = None) -> dict | None:
    updates = []
    params: list[Any] = []
    if title is not None:
        updates.append("title = ?")
        params.append(title.strip()[:80] or "新对话")
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if not updates:
        return get_conversation(conversation_id)
    updates.append("updated_at = ?")
    params.append(local_time_string())
    params.append(conversation_id)
    with sqlite3.connect(connection.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(f"UPDATE assistant_conversations SET {', '.join(updates)} WHERE id = ?", params)
        row = conn.execute("SELECT * FROM assistant_conversations WHERE id = ?", (conversation_id,)).fetchone()
        conn.commit()
        return dict(row) if row else None
