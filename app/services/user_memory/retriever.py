from __future__ import annotations

import os
import json
import math
from typing import Any

from app.core.db.connection import connect
from app.services.user_memory.catalog import allowed_predicates_for_route, query_terms
from app.services.user_memory.store import increment_usage


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else -1.0


def _should_use_dense(owner_id: str, workspace_id: str) -> bool:
    recall = os.getenv("USER_MEMORY_EVAL_RECALL_AT_5", "").strip()
    if recall:
        try:
            if float(recall) < 0.90:
                return True
        except ValueError:
            pass
    with connect(row_factory=True) as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM user_memories WHERE owner_id = ? AND workspace_id = ? AND status = 'active'",
            (owner_id, workspace_id),
        ).fetchone()["n"]
    return int(count) > 100


def _dense_rank(*, owner_id: str, workspace_id: str, query: str, allowed: set[str]) -> list[int]:
    """Best-effort dense rank using a Memory-only embedding table."""
    if not _should_use_dense(owner_id, workspace_id):
        return []
    try:
        from app.services.rag.embedding_service import embed_query, embed_texts, get_embedding_config

        model = get_embedding_config().model
        placeholders = ",".join("?" for _ in allowed)
        with connect(row_factory=True) as conn:
            memories = conn.execute(
                f"""
                SELECT id, search_text FROM user_memories
                WHERE owner_id = ? AND workspace_id = ? AND status = 'active'
                  AND predicate IN ({placeholders})
                ORDER BY updated_at DESC LIMIT 500
                """,
                (owner_id, workspace_id, *sorted(allowed)),
            ).fetchall()
            existing = {
                int(row["memory_id"]): json.loads(bytes(row["vector_blob"]).decode("utf-8"))
                for row in conn.execute(
                    f"""
                    SELECT e.memory_id, e.vector_blob FROM user_memory_embeddings e
                    JOIN user_memories m ON m.id = e.memory_id
                    WHERE m.owner_id = ? AND m.workspace_id = ? AND m.status = 'active'
                      AND e.embedding_model = ?
                      AND m.predicate IN ({placeholders})
                    """,
                    (owner_id, workspace_id, model, *sorted(allowed)),
                ).fetchall()
            }
        missing = [row for row in memories if int(row["id"]) not in existing]
        if missing:
            vectors = embed_texts([str(row["search_text"]) for row in missing])
            from app.core.time_utils import local_time_string
            from app.services.user_memory.catalog import value_hash
            now = local_time_string()
            with connect() as conn:
                for row, vector in zip(missing, vectors):
                    blob = json.dumps(vector, separators=(",", ":")).encode("utf-8")
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO user_memory_embeddings
                        (memory_id, content_hash, embedding_model, dimension, vector_blob, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (row["id"], value_hash(row["search_text"]), model, len(vector), blob, now, now),
                    )
                    existing[int(row["id"])] = vector
                conn.commit()
        query_vector = embed_query(query)
        return [memory_id for memory_id, _ in sorted(
            ((memory_id, _cosine(query_vector, vector)) for memory_id, vector in existing.items()),
            key=lambda pair: pair[1], reverse=True,
        )[:20]]
    except Exception:
        return []


def _rrf_merge(rows: list[Any], dense_ids: list[int]) -> list[Any]:
    if not dense_ids:
        return rows
    row_by_id = {int(row["id"]): row for row in rows}
    missing = [memory_id for memory_id in dense_ids if memory_id not in row_by_id]
    if missing:
        placeholders = ",".join("?" for _ in missing)
        with connect(row_factory=True) as conn:
            for row in conn.execute(f"SELECT *, 0.0 AS lexical_rank FROM user_memories WHERE id IN ({placeholders})", missing).fetchall():
                row_by_id[int(row["id"])] = row
    scores: dict[int, float] = {}
    for rank, row in enumerate(rows[:20], start=1):
        scores[int(row["id"])] = scores.get(int(row["id"]), 0.0) + 1.0 / (60 + rank)
    for rank, memory_id in enumerate(dense_ids[:20], start=1):
        scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (60 + rank)
    return [row_by_id[memory_id] for memory_id in sorted(scores, key=scores.get, reverse=True) if memory_id in row_by_id]


def _decode_row(row: Any) -> dict[str, Any]:
    import json

    item = dict(row)
    item["value"] = json.loads(item.pop("value_json"))
    return item


def retrieve_user_memories(*, owner_id: str, workspace_id: str, query: str,
                           route_kind: str | None = None, limit: int | None = None,
                           char_budget: int | None = None, record_usage: bool = True) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit or os.getenv("USER_MEMORY_MAX_ITEMS", "5")), 20))
    char_budget = max(100, int(char_budget or os.getenv("USER_MEMORY_CHAR_BUDGET", "1000")))
    allowed = allowed_predicates_for_route(route_kind)
    terms = query_terms(query)
    rows: list[Any] = []
    if terms:
        expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        placeholders = ",".join("?" for _ in allowed)
        with connect(row_factory=True) as conn:
            try:
                rows = conn.execute(
                    f"""
                    SELECT m.*, bm25(user_memories_fts) AS lexical_rank
                    FROM user_memories_fts
                    JOIN user_memories AS m ON m.id = CAST(user_memories_fts.memory_id AS INTEGER)
                    WHERE user_memories_fts MATCH ?
                      AND m.owner_id = ? AND m.workspace_id = ? AND m.status = 'active'
                      AND m.predicate IN ({placeholders})
                    ORDER BY lexical_rank ASC, m.confidence DESC, m.updated_at DESC
                    LIMIT ?
                    """,
                    (expression, owner_id, workspace_id, *sorted(allowed), max(limit * 4, 20)),
                ).fetchall()
            except Exception:
                rows = []
    if not rows:
        placeholders = ",".join("?" for _ in allowed)
        with connect(row_factory=True) as conn:
            rows = conn.execute(
                f"""
                SELECT *, 0.0 AS lexical_rank FROM user_memories
                WHERE owner_id = ? AND workspace_id = ? AND status = 'active'
                  AND predicate IN ({placeholders})
                ORDER BY confirmation_count DESC, confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (owner_id, workspace_id, *sorted(allowed), max(limit * 4, 20)),
            ).fetchall()

    rows = _rrf_merge(rows, _dense_rank(
        owner_id=owner_id, workspace_id=workspace_id, query=query, allowed=allowed,
    ))

    selected: list[dict[str, Any]] = []
    used = 0
    for index, row in enumerate(rows):
        item = _decode_row(row)
        rendered = f"{item['predicate']}: {item['value']}"
        if selected and used + len(rendered) > char_budget:
            break
        item["score"] = round(max(0.0, 1.0 - index / max(1, len(rows))) * float(item.get("confidence") or 0.0), 4)
        selected.append({
            "id": item["id"], "memory_type": item["memory_type"], "predicate": item["predicate"],
            "value": item["value"], "revision": item["revision"], "confidence": item["confidence"],
            "score": item["score"],
        })
        used += len(rendered)
        if len(selected) >= limit:
            break
    if record_usage:
        increment_usage([int(item["id"]) for item in selected], owner_id=owner_id, workspace_id=workspace_id, query=query)
    return selected
