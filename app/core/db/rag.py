import json
import sqlite3
import hashlib
import math
import os
import struct
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.db.connection import DB_PATH
from app.core.time_utils import local_time_string
from app.models.rag_evidence_schema import stable_json_dumps


DOCUMENT_STATUSES = {
    "uploaded", "indexing", "active", "pending", "indexed", "failed", "skipped", "archived",
    "review_required", "quarantined", "superseded",
}
RAG_INDEX_SCHEMA_VERSION = "2026-07-generation-v1"


def canonicalize_rag_source_path(source_path: str) -> str:
    resolved = os.path.normcase(os.path.abspath(os.path.normpath(str(source_path))))
    return resolved.replace("\\", "/")


def get_active_rag_generation_id(conn: sqlite3.Connection | None = None) -> str:
    owns_connection = conn is None
    connection = conn or sqlite3.connect(DB_PATH)
    try:
        row = connection.execute(
            "SELECT generation_id FROM rag_index_generations WHERE status = 'active' LIMIT 1"
        ).fetchone()
        return str(row[0]) if row else "legacy"
    finally:
        if owns_connection:
            connection.close()


def create_rag_index_generation(
    *,
    generation_id: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    reranker_model: str | None = None,
    parser_policy_version: str | None = None,
    source_snapshot_hash: str | None = None,
) -> dict:
    generation_id = generation_id or f"rag-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO rag_index_generations
            (generation_id, status, schema_version, parser_policy_version,
             embedding_model, embedding_dimension, reranker_model,
             source_snapshot_hash, health_json, metrics_json, created_at, updated_at)
            VALUES (?, 'building', ?, ?, ?, ?, ?, ?, '{}', '{}', ?, ?)
            """,
            (
                generation_id, RAG_INDEX_SCHEMA_VERSION, parser_policy_version,
                embedding_model, embedding_dimension, reranker_model,
                source_snapshot_hash, now, now,
            ),
        )
        conn.commit()
    return get_rag_index_generation(generation_id) or {"generation_id": generation_id, "status": "building"}


def get_rag_index_generation(generation_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM rag_index_generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
    return _generation_row(row) if row else None


def list_rag_index_generations() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM rag_index_generations ORDER BY created_at DESC"
        ).fetchall()
    return [_generation_row(row) for row in rows]


def update_rag_index_generation(
    generation_id: str,
    *,
    status: str | None = None,
    health: dict | None = None,
    metrics: dict | None = None,
    error_message: str | None = None,
    embedding_dimension: int | None = None,
) -> bool:
    if status is not None and status not in {"building", "ready", "failed", "retired"}:
        raise ValueError("invalid_generation_status")
    now = local_time_string()
    assignments = ["updated_at = ?"]
    params: list[Any] = [now]
    if status is not None:
        assignments.append("status = ?")
        params.append(status)
        if status == "ready":
            assignments.append("ready_at = ?")
            params.append(now)
    if health is not None:
        assignments.append("health_json = ?")
        params.append(json.dumps(health, ensure_ascii=False))
    if metrics is not None:
        assignments.append("metrics_json = ?")
        params.append(json.dumps(metrics, ensure_ascii=False))
    if error_message is not None:
        assignments.append("error_message = ?")
        params.append(error_message[:2000])
    if embedding_dimension is not None:
        assignments.append("embedding_dimension = ?")
        params.append(int(embedding_dimension))
    params.append(generation_id)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            f"UPDATE rag_index_generations SET {', '.join(assignments)} WHERE generation_id = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0


def activate_rag_index_generation(generation_id: str) -> bool:
    """Atomically switch readers; activating a retired healthy generation performs rollback."""
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        target = conn.execute(
            "SELECT status, health_json FROM rag_index_generations WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()
        if not target or target["status"] not in {"ready", "retired"}:
            return False
        try:
            health = json.loads(target["health_json"] or "{}")
        except Exception:
            health = {}
        if health and not health.get("activation_ready", False):
            return False
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE rag_index_generations SET status='retired', retired_at=?, updated_at=? WHERE status='active'",
            (now, now),
        )
        conn.execute(
            """
            UPDATE rag_index_generations
            SET status='active', activated_at=?, retired_at=NULL, updated_at=?
            WHERE generation_id=?
            """,
            (now, now, generation_id),
        )
        conn.commit()
    return True


def _generation_row(row: sqlite3.Row) -> dict:
    result = dict(row)
    for source, target in (("health_json", "health"), ("metrics_json", "metrics")):
        try:
            result[target] = json.loads(result.pop(source) or "{}")
        except Exception:
            result[target] = {}
    return result


def get_rag_document_by_path(source_path: str, generation_id: str | None = None) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        generation_id = generation_id or get_active_rag_generation_id(conn)
        canonical = _canonical_source_path(source_path)
        cursor.execute(
            """
            SELECT * FROM rag_documents
            WHERE generation_id = ? AND (canonical_source_path = ? OR source_path = ?)
            ORDER BY id DESC LIMIT 1
            """,
            (generation_id, canonical, canonical),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def consolidate_rag_document_path(source_path: str, generation_id: str | None = None) -> dict | None:
    """Merge legacy rows that point to the same physical source path."""
    canonical = _canonical_source_path(source_path)
    canonical_key = canonical.casefold()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        generation_id = generation_id or get_active_rag_generation_id(conn)
        cursor.execute("SELECT * FROM rag_documents WHERE generation_id = ? ORDER BY id", (generation_id,))
        matches = [
            dict(row)
            for row in cursor.fetchall()
            if _canonical_source_path(row.get("canonical_source_path") or row["source_path"]).casefold() == canonical_key
        ]
        if not matches:
            return None

        survivor = matches[0]
        duplicate_ids = [row["id"] for row in matches if row["id"] != survivor["id"]]
        for document_id in duplicate_ids:
            cursor.execute("DELETE FROM rag_chunks_fts WHERE document_id = ?", (document_id,))
            for table in (
                "rag_chunks",
                "rag_source_schemas",
                "rag_recipe_components",
                "rag_sop_facts",
                "rag_paper_facts",
                "rag_experiment_data_values",
            ):
                cursor.execute(f"DELETE FROM {table} WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM rag_documents WHERE id = ?", (document_id,))

        cursor.execute(
            "UPDATE rag_documents SET canonical_source_path = ? WHERE id = ?",
            (canonical, survivor["id"]),
        )
        conn.commit()
        cursor.execute("SELECT * FROM rag_documents WHERE id = ?", (survivor["id"],))
        row = cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        result["_duplicates_merged"] = bool(duplicate_ids)
        return result


def _canonical_source_path(source_path: str) -> str:
    return canonicalize_rag_source_path(source_path)


def _stable_source_id(source_path: str) -> str:
    digest = hashlib.sha256(_canonical_source_path(source_path).encode("utf-8")).hexdigest()[:16]
    return f"source:{digest}"


def upsert_rag_knowledge_source(
    source_path: str,
    doc_type: str,
    file_name: str | None = None,
    source_type: str = "local_file",
    version: str | None = None,
    year: str | None = None,
    language: str | None = None,
    trust_level: str = "lab_internal",
    owner: str | None = None,
    ingestion_status: str = "pending",
    content_hash: str | None = None,
    metadata: dict | None = None,
    chunk_count: int | None = None,
    last_error: str | None = None,
    asset_key: str | None = None,
    effective_from: str | None = None,
    effective_to: str | None = None,
    supersedes_source_id: str | None = None,
    security_flags: list[str] | None = None,
    review_status: str = "approved",
) -> str:
    canonical_path = _canonical_source_path(source_path)
    source_id = _stable_source_id(canonical_path)
    topic = str((metadata or {}).get("topic") or Path(source_path).stem)
    asset_key = asset_key or f"{doc_type}:{_asset_key_part(topic)}"
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for existing in conn.execute("SELECT source_id, source_path FROM rag_knowledge_sources"):
            if _canonical_source_path(existing["source_path"]).casefold() == canonical_path.casefold():
                source_id = str(existing["source_id"])
                break
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rag_knowledge_sources
            (source_id, source_type, doc_type, source_path, file_name, version, year,
             language, trust_level, owner, ingestion_status, content_hash, metadata_json,
             created_at, updated_at, chunk_count, last_error, canonical_source_path,
             asset_key, effective_from, effective_to, supersedes_source_id,
             security_flags_json, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                source_type = excluded.source_type,
                doc_type = excluded.doc_type,
                source_path = excluded.source_path,
                file_name = excluded.file_name,
                version = excluded.version,
                year = excluded.year,
                language = excluded.language,
                trust_level = excluded.trust_level,
                owner = excluded.owner,
                ingestion_status = excluded.ingestion_status,
                content_hash = excluded.content_hash,
                metadata_json = excluded.metadata_json,
                chunk_count = COALESCE(excluded.chunk_count, rag_knowledge_sources.chunk_count),
                last_error = excluded.last_error,
                canonical_source_path = excluded.canonical_source_path,
                asset_key = excluded.asset_key,
                effective_from = excluded.effective_from,
                effective_to = excluded.effective_to,
                supersedes_source_id = excluded.supersedes_source_id,
                security_flags_json = excluded.security_flags_json,
                review_status = excluded.review_status,
                updated_at = excluded.updated_at
            """,
            (
                source_id,
                source_type,
                doc_type,
                canonical_path,
                file_name or Path(source_path).name,
                version,
                year,
                language,
                trust_level,
                owner,
                ingestion_status,
                content_hash,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
                now,
                chunk_count,
                last_error,
                canonical_path,
                asset_key,
                effective_from,
                effective_to,
                supersedes_source_id,
                json.dumps(security_flags or [], ensure_ascii=False),
                review_status,
            ),
        )
        conn.commit()
    return source_id


def _asset_key_part(value: str) -> str:
    normalized = "_".join(str(value or "document").strip().casefold().split())
    return "".join(ch for ch in normalized if ch.isalnum() or ch in {"_", "-"}) or "document"


def get_rag_knowledge_source_by_path(source_path: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        canonical = _canonical_source_path(source_path)
        cursor.execute(
            "SELECT * FROM rag_knowledge_sources WHERE canonical_source_path = ? OR source_path = ?",
            (canonical, canonical),
        )
        row = cursor.fetchone()
        if not row:
            cursor.execute("SELECT * FROM rag_knowledge_sources")
            row = next(
                (
                    item for item in cursor.fetchall()
                    if _canonical_source_path(item["source_path"]).casefold() == canonical.casefold()
                ),
                None,
            )
    if not row:
        return None
    return _decode_source_row(row)


def list_rag_knowledge_sources(doc_type: str | None = None) -> list[dict]:
    params: list[Any] = []
    where = ""
    if doc_type:
        where = "WHERE doc_type = ?"
        params.append(doc_type)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT * FROM rag_knowledge_sources {where} ORDER BY doc_type, file_name",
            params,
        )
        rows = [dict(row) for row in cursor.fetchall()]
    return [_decode_source_row(row) for row in rows]


def get_rag_knowledge_source(source_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM rag_knowledge_sources WHERE source_id = ?", (source_id,)
        ).fetchone()
    if not row:
        return None
    return _decode_source_row(row)


def _decode_source_row(row: sqlite3.Row | dict) -> dict:
    result = dict(row)
    try:
        result["metadata"] = json.loads(result.pop("metadata_json") or "{}")
    except Exception:
        result["metadata"] = {}
    try:
        result["security_flags"] = json.loads(result.pop("security_flags_json") or "[]")
    except Exception:
        result["security_flags"] = []
    return result


def set_rag_knowledge_source_lifecycle(source_id: str, status: str, *, actor: str | None = None) -> bool:
    if status not in {"active", "archived", "superseded", "quarantined", "review_required"}:
        raise ValueError("invalid_knowledge_lifecycle")
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT source_path, ingestion_status FROM rag_knowledge_sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        if not row:
            return False
        conn.execute(
            """
            UPDATE rag_knowledge_sources
            SET lifecycle_status = ?, archived_at = ?, archived_by = ?, updated_at = ?
            WHERE source_id = ?
            """,
            (status, now if status == "archived" else None, actor if status == "archived" else None, now, source_id),
        )
        document_status = (
            row["ingestion_status"] if status == "active" and row["ingestion_status"] in {"indexed", "skipped"}
            else status
        )
        conn.execute(
            "UPDATE rag_documents SET status = ?, updated_at = ? WHERE source_id = ? OR canonical_source_path = ?",
            (document_status, now, source_id, _canonical_source_path(row["source_path"])),
        )
        conn.commit()
        return True


def list_rag_documents(status: str | None = None, generation_id: str | None = None) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        generation_id = generation_id or get_active_rag_generation_id(conn)
        if status:
            cursor.execute(
                "SELECT * FROM rag_documents WHERE generation_id = ? AND status = ? ORDER BY doc_type, file_name",
                (generation_id, status),
            )
        else:
            cursor.execute(
                "SELECT * FROM rag_documents WHERE generation_id = ? ORDER BY doc_type, file_name",
                (generation_id,),
            )
        return [dict(row) for row in cursor.fetchall()]


def upsert_rag_document(
    source_path: str,
    file_name: str,
    doc_type: str,
    topic: str | None,
    version: str | None,
    year: str | None,
    language: str | None,
    content_hash: str,
    status: str = "pending",
    error_message: str | None = None,
    generation_id: str | None = None,
    source_id: str | None = None,
) -> int:
    if status not in DOCUMENT_STATUSES:
        status = "pending"
    now = local_time_string()
    canonical_path = _canonical_source_path(source_path)
    with sqlite3.connect(DB_PATH) as conn:
        generation_id = generation_id or get_active_rag_generation_id(conn)
        stored_path = canonical_path if generation_id == "legacy" else f"{canonical_path}#rag-generation={generation_id}"
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rag_documents
            (source_path, file_name, doc_type, topic, version, year, language,
             content_hash, status, error_message, created_at, updated_at,
             generation_id, source_id, canonical_source_path, index_schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                file_name = excluded.file_name,
                doc_type = excluded.doc_type,
                topic = excluded.topic,
                version = excluded.version,
                year = excluded.year,
                language = excluded.language,
                content_hash = excluded.content_hash,
                status = excluded.status,
                error_message = excluded.error_message,
                generation_id = excluded.generation_id,
                source_id = excluded.source_id,
                canonical_source_path = excluded.canonical_source_path,
                index_schema_version = excluded.index_schema_version,
                updated_at = excluded.updated_at
            """,
            (
                stored_path, file_name, doc_type, topic, version, year, language,
                content_hash, status, error_message, now, now, generation_id,
                source_id, canonical_path, RAG_INDEX_SCHEMA_VERSION,
            ),
        )
        conn.commit()
        cursor.execute("SELECT id FROM rag_documents WHERE source_path = ?", (stored_path,))
        return int(cursor.fetchone()[0])


def mark_rag_document_indexed(document_id: int, chunk_count: int) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE rag_documents
            SET status = ?, indexed_at = ?, chunk_count = ?, error_message = NULL, updated_at = ?
            WHERE id = ?
            """,
            ("indexed", now, int(chunk_count), now, document_id),
        )
        conn.commit()


def mark_rag_document_skipped(document_id: int) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE rag_documents SET status = ?, updated_at = ? WHERE id = ?",
            ("skipped", now, document_id),
        )
        conn.commit()


def mark_rag_document_failed(document_id: int, error_message: str) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE rag_documents
            SET status = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            ("failed", error_message[:1000], now, document_id),
        )
        conn.commit()


def replace_rag_chunks(document_id: int, chunks: list[dict]) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id FROM rag_chunks WHERE document_id = ?", (document_id,))
        old_chunk_ids = [row[0] for row in cursor.fetchall()]
        if old_chunk_ids:
            placeholders = ",".join("?" for _ in old_chunk_ids)
            cursor.execute(
                f"DELETE FROM rag_chunks_fts WHERE chunk_id IN ({placeholders})",
                old_chunk_ids,
            )
            cursor.execute(
                f"DELETE FROM rag_chunk_embeddings WHERE chunk_id IN ({placeholders})",
                old_chunk_ids,
            )
        cursor.execute("DELETE FROM rag_chunks WHERE document_id = ?", (document_id,))

        for chunk in chunks:
            cursor.execute(
                """
                INSERT INTO rag_chunks
                (document_id, chunk_id, doc_type, source_path, file_name, title,
                 section, page_number, sheet_name, row_start, row_end, chunk_index,
                 content, content_hash, metadata_json, created_at, generation_id,
                 context_prefix, index_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk["chunk_id"],
                    chunk["doc_type"],
                    chunk["source_path"],
                    chunk["file_name"],
                    chunk.get("title"),
                    chunk.get("section"),
                    chunk.get("page_number"),
                    chunk.get("sheet_name"),
                    chunk.get("row_start"),
                    chunk.get("row_end"),
                    int(chunk["chunk_index"]),
                    chunk["content"],
                    chunk["content_hash"],
                    json.dumps(chunk.get("metadata") or {}, ensure_ascii=False),
                    now,
                    chunk.get("generation_id") or "legacy",
                    chunk.get("context_prefix") or "",
                    chunk.get("index_text") or chunk["content"],
                ),
            )
            cursor.execute(
                """
                INSERT INTO rag_chunks_fts
                (chunk_id, document_id, doc_type, file_name, title, section, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"],
                    document_id,
                    chunk["doc_type"],
                    chunk["file_name"],
                    chunk.get("title") or "",
                    chunk.get("section") or "",
                    chunk.get("index_text") or chunk["content"],
                ),
            )
        conn.commit()


def replace_rag_chunk_embeddings(
    document_id: int,
    embeddings: list[dict],
    embedding_model: str,
    vector_backend: str = "sqlite_blob_fallback",
) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        generation_row = cursor.execute(
            "SELECT generation_id FROM rag_documents WHERE id = ?", (document_id,)
        ).fetchone()
        generation_id = str(generation_row[0]) if generation_row else "legacy"
        cursor.execute("DELETE FROM rag_chunk_embeddings WHERE document_id = ?", (document_id,))
        sqlite_vec_rows = []
        for item in embeddings:
            vector = _normalize_vector(item.get("vector") or [])
            if not vector:
                continue
            cursor.execute(
                """
                INSERT OR REPLACE INTO rag_chunk_embeddings
                (chunk_id, document_id, content_hash, embedding_model, dimension,
                 vector_blob, vector_backend, created_at, updated_at, generation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["chunk_id"],
                    document_id,
                    item["content_hash"],
                    embedding_model,
                    len(vector),
                    _pack_vector(vector),
                    vector_backend,
                    now,
                    now,
                    generation_id,
                ),
            )
            sqlite_vec_rows.append((int(cursor.lastrowid), vector))
        if vector_backend == "sqlite_vec" and not _try_sync_sqlite_vec(conn, sqlite_vec_rows):
            cursor.execute(
                "UPDATE rag_chunk_embeddings SET vector_backend = ? WHERE document_id = ?",
                ("sqlite_blob_fallback", document_id),
            )
        conn.commit()


def list_rag_chunk_embeddings(
    document_id: int | None = None,
    embedding_model: str | None = None,
    generation_id: str | None = None,
) -> list[dict]:
    params: list[Any] = []
    filters = []
    if document_id is not None:
        filters.append("document_id = ?")
        params.append(document_id)
    if embedding_model:
        filters.append("embedding_model = ?")
        params.append(embedding_model)
    if generation_id:
        filters.append("generation_id = ?")
        params.append(generation_id)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT id, chunk_id, document_id, content_hash, embedding_model,
                   dimension, vector_backend, created_at, updated_at
            FROM rag_chunk_embeddings
            {where}
            ORDER BY document_id, chunk_id
            """,
            params,
        )
        return [dict(row) for row in cursor.fetchall()]


def search_rag_chunk_embeddings(
    query_vector: list[float],
    embedding_model: str,
    top_k: int = 5,
    doc_types: list[str] | None = None,
    limit: int = 1000,
    generation_id: str | None = None,
    source_ids: list[str] | None = None,
    version_policy: str = "current",
    as_of: str | None = None,
) -> list[dict]:
    vector = _normalize_vector(query_vector)
    if not vector:
        return []
    generation_id = generation_id or get_active_rag_generation_id()
    sqlite_vec_rows = _try_search_sqlite_vec(
        vector, embedding_model, top_k, doc_types, generation_id, source_ids, version_policy, as_of
    )
    if sqlite_vec_rows is not None:
        return sqlite_vec_rows
    params: list[Any] = [embedding_model, generation_id]
    doc_type_filter = ""
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        doc_type_filter = f" AND c.doc_type IN ({placeholders})"
        params.extend(doc_types)
    source_scope, source_params = _rag_source_scope_sql(source_ids, version_policy, as_of)
    params.extend(source_params)
    params.append(max(int(limit or 1000), 1))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                e.vector_blob,
                e.dimension,
                e.embedding_model,
                e.vector_backend,
                c.*,
                d.topic,
                d.version,
                d.year,
                d.language
            FROM rag_chunk_embeddings e
            JOIN rag_chunks c ON c.chunk_id = e.chunk_id
            JOIN rag_documents d ON d.id = c.document_id
            JOIN rag_knowledge_sources s ON s.source_id = d.source_id
            WHERE e.embedding_model = ? AND d.generation_id = ?
              AND d.status IN ('indexed', 'skipped') {doc_type_filter} {source_scope}
            ORDER BY d.doc_type, d.file_name, c.chunk_index
            LIMIT ?
            """,
            params,
        )
        rows = [dict(row) for row in cursor.fetchall()]

    scored = []
    for row in rows:
        stored_vector = _unpack_vector(row.pop("vector_blob"), int(row.pop("dimension") or 0))
        score = _cosine(vector, stored_vector)
        if score <= 0:
            continue
        try:
            row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
        except Exception:
            row["metadata"] = {}
        row["vector_score"] = score
        row["embedding_model"] = embedding_model
        scored.append(row)
    scored.sort(
        key=lambda row: (
            -float(row.get("vector_score") or 0.0),
            row.get("file_name") or "",
            int(row.get("chunk_index") or 0),
        )
    )
    return scored[: max(int(top_k or 5), 1)]


def _try_sync_sqlite_vec(conn: sqlite3.Connection, rows: list[tuple[int, list[float]]]) -> bool:
    if not rows:
        return False
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        dimension = len(rows[0][1])
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunk_embedding_vec USING vec0(embedding float[{dimension}])"
        )
        for rowid, vector in rows:
            cursor.execute(
                "INSERT OR REPLACE INTO rag_chunk_embedding_vec(rowid, embedding) VALUES (?, ?)",
                (rowid, json.dumps(vector)),
            )
        return True
    except Exception:
        return False


def _try_search_sqlite_vec(
    query_vector: list[float],
    embedding_model: str,
    top_k: int,
    doc_types: list[str] | None,
    generation_id: str,
    source_ids: list[str] | None,
    version_policy: str,
    as_of: str | None,
) -> list[dict] | None:
    try:
        import sqlite_vec
    except Exception:
        return None
    params: list[Any] = [json.dumps(query_vector), max(int(top_k or 5), 1), embedding_model, generation_id]
    doc_type_filter = ""
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        doc_type_filter = f" AND c.doc_type IN ({placeholders})"
        params.extend(doc_types)
    source_scope, source_params = _rag_source_scope_sql(source_ids, version_policy, as_of)
    params.extend(source_params)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT
                    v.distance,
                    e.embedding_model,
                    e.vector_backend,
                    c.*,
                    d.topic,
                    d.version,
                    d.year,
                    d.language
                FROM rag_chunk_embedding_vec v
                JOIN rag_chunk_embeddings e ON e.id = v.rowid
                JOIN rag_chunks c ON c.chunk_id = e.chunk_id
                JOIN rag_documents d ON d.id = c.document_id
                JOIN rag_knowledge_sources s ON s.source_id = d.source_id
                WHERE v.embedding MATCH ? AND k = ?
                  AND e.embedding_model = ? AND d.generation_id = ?
                  AND d.status IN ('indexed', 'skipped') {doc_type_filter} {source_scope}
                ORDER BY v.distance
                """,
                params,
            )
            rows = [dict(row) for row in cursor.fetchall()]
    except Exception:
        return None
    results = []
    for row in rows:
        distance = float(row.pop("distance") or 0.0)
        try:
            row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
        except Exception:
            row["metadata"] = {}
        row["vector_score"] = 1.0 / (1.0 + max(distance, 0.0))
        results.append(row)
    return results


def _pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *[float(item) for item in vector])


def _unpack_vector(blob: bytes | None, dimension: int) -> list[float]:
    if not blob or dimension <= 0:
        return []
    try:
        return list(struct.unpack(f"<{dimension}f", blob))
    except struct.error:
        return []


def _normalize_vector(vector: list[float]) -> list[float]:
    values = [float(item) for item in vector if item is not None]
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0:
        return []
    return [item / norm for item in values]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def replace_rag_source_schemas(document_id: int, schema_rows: list[dict]) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rag_source_schemas WHERE document_id = ?", (document_id,))
        for row in schema_rows:
            cursor.execute(
                """
                INSERT INTO rag_source_schemas
                (document_id, source_file, source_path, sheet_name, column_name,
                 normalized_column_name, unit, inferred_role, column_index,
                 sample_values_json, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    row["source_file"],
                    row["source_path"],
                    row.get("sheet_name"),
                    row["column_name"],
                    row.get("normalized_column_name"),
                    row.get("unit"),
                    row.get("inferred_role"),
                    row.get("column_index"),
                    json.dumps(row.get("sample_values") or [], ensure_ascii=False),
                    float(row.get("confidence", 1.0)),
                    now,
                ),
            )
        conn.commit()


def replace_rag_recipe_components(document_id: int, component_rows: list[dict]) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rag_recipe_components WHERE document_id = ?", (document_id,))
        for row in component_rows:
            cursor.execute(
                """
                INSERT INTO rag_recipe_components
                (document_id, source_file, source_path, entity, group_name,
                 component_name, normalized_component_name, amount, unit, amount_text,
                 solution_type, working_addition, section, table_index, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    row["source_file"],
                    row["source_path"],
                    row.get("entity"),
                    row.get("group_name"),
                    row["component_name"],
                    row.get("normalized_component_name"),
                    row.get("amount"),
                    row.get("unit"),
                    row.get("amount_text"),
                    row.get("solution_type"),
                    row.get("working_addition"),
                    row.get("section"),
                    row.get("table_index"),
                    float(row.get("confidence", 1.0)),
                    now,
                ),
            )
        conn.commit()


def replace_rag_sop_facts(document_id: int, fact_rows: list[dict]) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rag_sop_facts WHERE document_id = ?", (document_id,))
        for row in fact_rows:
            cursor.execute(
                """
                INSERT INTO rag_sop_facts
                (document_id, source_file, source_path, entity, attribute, value,
                 text_span, section, page_number, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    row["source_file"],
                    row["source_path"],
                    row.get("entity"),
                    row.get("attribute"),
                    row.get("value"),
                    row.get("text_span"),
                    row.get("section"),
                    row.get("page_number"),
                    float(row.get("confidence", 1.0)),
                    now,
                ),
            )
        conn.commit()


def replace_rag_paper_facts(document_id: int, fact_rows: list[dict]) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rag_paper_facts WHERE document_id = ?", (document_id,))
        for row in fact_rows:
            cursor.execute(
                """
                INSERT INTO rag_paper_facts
                (document_id, source_file, source_path, paper_title, year, entity,
                 attribute, value, text_span, page_number, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    row["source_file"],
                    row["source_path"],
                    row.get("paper_title"),
                    row.get("year"),
                    row.get("entity"),
                    row.get("attribute"),
                    row.get("value"),
                    row.get("text_span"),
                    row.get("page_number"),
                    float(row.get("confidence", 1.0)),
                    now,
                ),
            )
        conn.commit()


def replace_rag_experiment_data_values(document_id: int, value_rows: list[dict]) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rag_experiment_data_values WHERE document_id = ?", (document_id,))
        for row in value_rows:
            cursor.execute(
                """
                INSERT INTO rag_experiment_data_values
                (document_id, source_file, source_path, sheet_name, row_index, column_name,
                 normalized_column_name, inferred_role, value_text, numeric_value, unit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    row["source_file"],
                    row["source_path"],
                    row.get("sheet_name"),
                    row.get("row_index"),
                    row["column_name"],
                    row.get("normalized_column_name"),
                    row.get("inferred_role"),
                    row.get("value_text"),
                    row.get("numeric_value"),
                    row.get("unit"),
                    now,
                ),
            )
        conn.commit()


def replace_rag_evidence_units(
    document_id: int,
    source_id: str | None,
    evidence_units: list[Any],
) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        generation_row = cursor.execute(
            "SELECT generation_id FROM rag_documents WHERE id = ?", (document_id,)
        ).fetchone()
        generation_id = str(generation_row[0]) if generation_row else "legacy"
        cursor.execute("DELETE FROM rag_evidence_units WHERE document_id = ?", (document_id,))
        for unit in evidence_units:
            row = _evidence_unit_row(unit)
            location = row.get("location") or {}
            citation = row.get("citation") or {}
            metadata = row.get("metadata") or {}
            cursor.execute(
                """
                INSERT OR REPLACE INTO rag_evidence_units
                (document_id, document_version, source_id, evidence_id, evidence_type,
                 content, content_hash, source_type, source_file, entity, relation,
                 target_entity, attribute, value, unit, text_span, page_number,
                 section, section_path, sheet_name, row_start, row_end, parent_id,
                 previous_id, next_id, source_locator, parser_version, element_type,
                 confidence, extraction_method, location_json, citation_json,
                 metadata_json, created_at, generation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("document_id") or document_id,
                    row.get("document_version") or metadata.get("version") or citation.get("version"),
                    row.get("source_id") or source_id,
                    row["evidence_id"],
                    row.get("evidence_type") or row.get("fact_type"),
                    row.get("content") or row.get("text_span") or row.get("value") or row.get("attribute") or "",
                    row.get("content_hash"),
                    row.get("source_type") or "document",
                    row.get("source_file") or "",
                    row.get("entity"),
                    row.get("relation"),
                    row.get("target_entity"),
                    row.get("attribute"),
                    row.get("value"),
                    row.get("unit"),
                    row.get("text_span"),
                    row.get("page_number") or location.get("page_number"),
                    row.get("section") or location.get("section"),
                    stable_json_dumps(row.get("section_path") or []),
                    row.get("sheet_name") or location.get("sheet_name"),
                    row.get("row_start") or location.get("row_start") or location.get("row_index"),
                    row.get("row_end") or location.get("row_end"),
                    row.get("parent_id") or metadata.get("parent_id"),
                    row.get("previous_id") or metadata.get("previous_id"),
                    row.get("next_id") or metadata.get("next_id"),
                    stable_json_dumps(row.get("source_locator") or {}),
                    row.get("parser_version") or metadata.get("parser_version"),
                    row.get("element_type") or metadata.get("element_type") or row.get("fact_type"),
                    float(row.get("confidence", 1.0)),
                    row.get("extraction_method") or metadata.get("extraction_method") or "adapter",
                    json.dumps(location, ensure_ascii=False),
                    json.dumps(citation, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    generation_id,
                ),
            )
        conn.commit()
    replace_rag_evidence_graph(evidence_units, generation_id=generation_id)


def _evidence_unit_row(unit: Any) -> dict:
    if hasattr(unit, "model_dump"):
        return unit.model_dump()
    return dict(unit)


def list_rag_evidence_units(
    evidence_types: list[str] | None = None,
    source_id: str | None = None,
    limit: int = 1000,
) -> list[dict]:
    params: list[Any] = [get_active_rag_generation_id()]
    filters = ["e.generation_id = ?"]
    if evidence_types:
        placeholders = ",".join("?" for _ in evidence_types)
        filters.append(f"e.evidence_type IN ({placeholders})")
        params.extend(evidence_types)
    if source_id:
        filters.append("e.source_id = ?")
        params.append(source_id)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(max(int(limit or 1000), 1))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT e.*
            FROM rag_evidence_units e
            LEFT JOIN rag_documents d ON d.id = e.document_id
            {where}
            {'AND' if where else 'WHERE'} (d.id IS NULL OR d.status IN ('indexed', 'skipped'))
            ORDER BY source_file, evidence_type, id
            LIMIT ?
            """,
            params,
        )
        rows = [dict(row) for row in cursor.fetchall()]
    for row in rows:
        for key in ("location_json", "citation_json", "metadata_json"):
            target_key = key.replace("_json", "")
            try:
                row[target_key] = json.loads(row.pop(key) or "{}")
            except Exception:
                row[target_key] = {}
        for key, fallback in (("section_path", []), ("source_locator", {})):
            raw_value = row.get(key)
            if raw_value in {None, ""}:
                row[key] = fallback
                continue
            try:
                row[key] = json.loads(raw_value)
            except Exception:
                row[key] = fallback
    return _filter_governed_rag_rows(rows)


def replace_rag_document_elements(document_id: str | int, elements: list[Any]) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rag_document_elements WHERE document_id = ?", (str(document_id),))
        for element in elements:
            row = element.model_dump() if hasattr(element, "model_dump") else dict(element)
            cursor.execute(
                """
                INSERT OR REPLACE INTO rag_document_elements
                (element_id, document_id, document_version, element_type, content,
                 page_number, section_path, parent_id, previous_id, next_id,
                 source_locator, metadata_json, content_hash, parser_name,
                 parser_version, created_at, generation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["element_id"],
                    str(row.get("document_id") or document_id),
                    row.get("document_version") or (row.get("metadata") or {}).get("document_version"),
                    row.get("element_type") or "unknown",
                    row.get("content") or "",
                    row.get("page_number"),
                    stable_json_dumps(row.get("section_path") or []),
                    row.get("parent_id"),
                    row.get("previous_id"),
                    row.get("next_id"),
                    stable_json_dumps(row.get("source_locator") or {}),
                    json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                    row.get("content_hash") or "",
                    row.get("parser_name") or (row.get("metadata") or {}).get("parser_name"),
                    row.get("parser_version") or (row.get("metadata") or {}).get("parser_version"),
                    now,
                    (row.get("metadata") or {}).get("generation_id") or "legacy",
                ),
            )
        conn.commit()


def replace_rag_hierarchy_nodes(document_id: int, generation_id: str, nodes: list[dict]) -> None:
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM rag_hierarchy_nodes WHERE document_id = ?", (document_id,))
        for node in nodes:
            conn.execute(
                """
                INSERT INTO rag_hierarchy_nodes
                (node_id, generation_id, document_id, level, parent_node_id, title, summary, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node["node_id"], generation_id, document_id, int(node["level"]),
                    node.get("parent_node_id"), node.get("title"), node.get("summary") or "",
                    json.dumps(node.get("metadata") or {}, ensure_ascii=False), now,
                ),
            )
        conn.commit()


def list_rag_document_elements(
    document_id: str | int | None = None,
    parent_id: str | None = None,
    limit: int = 5000,
) -> list[dict]:
    params: list[Any] = [get_active_rag_generation_id()]
    filters = ["generation_id = ?"]
    if document_id is not None:
        filters.append("document_id = ?")
        params.append(str(document_id))
    if parent_id:
        filters.append("parent_id = ?")
        params.append(parent_id)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(max(int(limit or 5000), 1))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT *
            FROM rag_document_elements
            {where}
            ORDER BY created_at, element_id
            LIMIT ?
            """,
            params,
        )
        rows = [dict(row) for row in cursor.fetchall()]
    for row in rows:
        for key, fallback in (("section_path", []), ("source_locator", {}), ("metadata_json", {})):
            raw_value = row.get(key)
            final_key = "metadata" if key == "metadata_json" else key
            try:
                row[final_key] = json.loads(raw_value or ("[]" if fallback == [] else "{}"))
            except Exception:
                row[final_key] = fallback
            if key == "metadata_json":
                row.pop("metadata_json", None)
    return rows


def insert_rag_trace_log(
    query_id: str,
    user_query: str,
    route: str | None = None,
    query_frame: dict | None = None,
    retrieval_channels: list[str] | None = None,
    retrieved_ids: list[str] | None = None,
    reranked_ids: list[str] | None = None,
    selected_evidence_ids: list[str] | None = None,
    answerability: dict | None = None,
    sufficiency: dict | None = None,
    citations: list[dict] | None = None,
    latency_ms: int | None = None,
    status: str | None = None,
) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO rag_trace_logs
            (query_id, user_query, route, query_frame_json, retrieval_channels_json,
             retrieved_ids_json, reranked_ids_json, selected_evidence_ids_json,
             answerability_json, sufficiency_json, citations_json, latency_ms,
             status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                user_query,
                route,
                json.dumps(query_frame or {}, ensure_ascii=False),
                json.dumps(retrieval_channels or [], ensure_ascii=False),
                json.dumps(retrieved_ids or [], ensure_ascii=False),
                json.dumps(reranked_ids or [], ensure_ascii=False),
                json.dumps(selected_evidence_ids or [], ensure_ascii=False),
                json.dumps(answerability or {}, ensure_ascii=False),
                json.dumps(sufficiency or {}, ensure_ascii=False),
                json.dumps(citations or [], ensure_ascii=False),
                latency_ms,
                status,
                local_time_string(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_rag_trace_logs(limit: int = 20) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM rag_trace_logs ORDER BY id DESC LIMIT ?",
            (max(int(limit or 20), 1),),
        )
        rows = [dict(row) for row in cursor.fetchall()]
    json_fields = {
        "query_frame_json": "query_frame",
        "retrieval_channels_json": "retrieval_channels",
        "retrieved_ids_json": "retrieved_ids",
        "reranked_ids_json": "reranked_ids",
        "selected_evidence_ids_json": "selected_evidence_ids",
        "answerability_json": "answerability",
        "sufficiency_json": "sufficiency",
        "citations_json": "citations",
    }
    for row in rows:
        for raw_key, final_key in json_fields.items():
            raw_value = row.pop(raw_key, None)
            try:
                row[final_key] = json.loads(raw_value or ("[]" if final_key in {"retrieval_channels", "retrieved_ids", "reranked_ids", "selected_evidence_ids", "citations"} else "{}"))
            except Exception:
                row[final_key] = [] if final_key.endswith("ids") or final_key == "citations" else {}
    return rows


def insert_rag_retrieval_trace_rows(query_id: str, rows: list[dict]) -> None:
    if not rows:
        return
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rag_retrieval_traces WHERE query_id = ?", (query_id,))
        for row in rows:
            cursor.execute(
                """
                INSERT INTO rag_retrieval_traces
                (query_id, stage, evidence_id, document_id, rank, sparse_score,
                dense_score, fusion_score, rerank_score, selected,
                 rejection_reason, metadata_json, created_at, generation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    row.get("stage"),
                    row.get("evidence_id"),
                    str(row.get("document_id")) if row.get("document_id") is not None else None,
                    row.get("rank"),
                    row.get("sparse_score"),
                    row.get("dense_score"),
                    row.get("fusion_score"),
                    row.get("rerank_score"),
                    1 if row.get("selected") else 0,
                    row.get("rejection_reason"),
                    json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                    now,
                    row.get("generation_id") or get_active_rag_generation_id(conn),
                ),
            )
        conn.commit()


def list_rag_retrieval_trace_rows(query_id: str | None = None, limit: int = 200) -> list[dict]:
    params: list[Any] = []
    where = ""
    if query_id:
        where = "WHERE query_id = ?"
        params.append(query_id)
    params.append(max(int(limit or 200), 1))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT *
            FROM rag_retrieval_traces
            {where}
            ORDER BY id
            LIMIT ?
            """,
            params,
        )
        rows = [dict(row) for row in cursor.fetchall()]
    for row in rows:
        try:
            row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
        except Exception:
            row["metadata"] = {}
        row["selected"] = bool(row.get("selected"))
    return rows


def upsert_rag_eval_cases(cases: list[dict]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for case in cases:
            cursor.execute(
                """
                INSERT OR REPLACE INTO rag_eval_cases
                (id, category, question, expected_route, required_evidence_types_json,
                 expected_answer_contains_json, expected_evidence_labels_json,
                 retrieval_mode, version_policy, as_of, expected_refusal,
                 expected_action_escalation, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case["id"],
                    case["category"],
                    case["question"],
                    case.get("expected_route"),
                    json.dumps(case.get("required_evidence_types") or [], ensure_ascii=False),
                    json.dumps(case.get("expected_answer_contains") or [], ensure_ascii=False),
                    json.dumps(case.get("expected_evidence_labels") or [], ensure_ascii=False),
                    case.get("retrieval_mode") or "auto",
                    case.get("version_policy") or "current",
                    case.get("as_of"),
                    1 if case.get("expected_refusal") else 0,
                    1 if case.get("expected_action_escalation") else 0,
                    case.get("notes"),
                ),
            )
        conn.commit()


def list_rag_eval_cases() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rag_eval_cases ORDER BY category, id")
        rows = [dict(row) for row in cursor.fetchall()]
    for row in rows:
        row["required_evidence_types"] = json.loads(row.pop("required_evidence_types_json") or "[]")
        row["expected_answer_contains"] = json.loads(row.pop("expected_answer_contains_json") or "[]")
        row["expected_evidence_labels"] = json.loads(row.pop("expected_evidence_labels_json") or "[]")
        row["expected_refusal"] = bool(row.get("expected_refusal"))
        row["expected_action_escalation"] = bool(row.get("expected_action_escalation"))
    return rows


def replace_rag_evidence_graph(evidence_units: list[Any], generation_id: str = "legacy") -> None:
    """Build a small SQLite graph from evidence; heavier GraphRAG can sit behind this later."""
    rows = [_evidence_unit_row(unit) for unit in evidence_units]
    now = local_time_string()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for row in rows:
            evidence_id = row.get("evidence_id")
            source_file = row.get("source_file") or "source"
            source_entity = _graph_entity_id(row.get("entity") or source_file)
            target_value = row.get("target_entity") or row.get("value") or row.get("attribute")
            if not evidence_id or not target_value:
                continue
            target_entity = _graph_entity_id(str(target_value))
            relation = row.get("relation") or _relation_for_evidence(row)
            cursor.execute(
                """
                INSERT OR IGNORE INTO rag_entities
                (entity_id, entity_type, canonical_name, aliases_json, metadata_json, created_at, generation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_entity,
                    row.get("source_type") or "unknown",
                    row.get("entity") or source_file,
                    "[]",
                    "{}",
                    now,
                    generation_id,
                ),
            )
            cursor.execute(
                """
                INSERT OR IGNORE INTO rag_entities
                (entity_id, entity_type, canonical_name, aliases_json, metadata_json, created_at, generation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (target_entity, row.get("evidence_type") or row.get("fact_type") or "evidence", str(target_value), "[]", "{}", now, generation_id),
            )
            cursor.execute(
                """
                INSERT INTO rag_relations
                (source_entity_id, relation, target_entity_id, evidence_id, confidence, created_at, generation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_entity,
                    relation,
                    target_entity,
                    evidence_id,
                    float(row.get("confidence", 1.0)),
                    now,
                    generation_id,
                ),
            )
            cursor.execute(
                """
                INSERT INTO rag_evidence_links
                (evidence_id, entity_id, role, created_at, generation_id)
                VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)
                """,
                (
                    evidence_id, source_entity, "source", now, generation_id,
                    evidence_id, target_entity, "target", now, generation_id,
                ),
            )
        conn.commit()


def list_rag_relations(entity_id: str | None = None) -> list[dict]:
    params: list[Any] = [get_active_rag_generation_id()]
    where = "WHERE generation_id = ?"
    if entity_id:
        where += " AND (source_entity_id = ? OR target_entity_id = ?)"
        params.extend([_graph_entity_id(entity_id), _graph_entity_id(entity_id)])
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT * FROM rag_relations {where} ORDER BY id",
            params,
        )
        return [dict(row) for row in cursor.fetchall()]


def list_rag_graph_neighbors(entity_id: str, depth: int = 1, limit: int = 50) -> dict:
    start = _graph_entity_id(entity_id)
    max_depth = max(1, min(int(depth or 1), 2))
    max_rows = max(1, int(limit or 50))
    visited = {start}
    frontier = {start}
    relations = []
    generation_id = get_active_rag_generation_id()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        for current_depth in range(1, max_depth + 1):
            if not frontier or len(relations) >= max_rows:
                break
            placeholders = ",".join("?" for _ in frontier)
            params = [generation_id, *frontier, *frontier, max_rows - len(relations)]
            cursor.execute(
                f"""
                SELECT
                    r.*,
                    s.canonical_name AS source_name,
                    s.entity_type AS source_type,
                    t.canonical_name AS target_name,
                    t.entity_type AS target_type
                FROM rag_relations r
                LEFT JOIN rag_entities s ON s.entity_id = r.source_entity_id
                LEFT JOIN rag_entities t ON t.entity_id = r.target_entity_id
                WHERE r.generation_id = ? AND (
                    r.source_entity_id IN ({placeholders})
                    OR r.target_entity_id IN ({placeholders})
                )
                ORDER BY r.id
                LIMIT ?
                """,
                params,
            )
            rows = [dict(row) for row in cursor.fetchall()]
            next_frontier = set()
            for row in rows:
                row["depth"] = current_depth
                relations.append(row)
                for key in ("source_entity_id", "target_entity_id"):
                    neighbor = row.get(key)
                    if neighbor and neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
    return {
        "start_entity_id": start,
        "depth": max_depth,
        "entity_count": len(visited),
        "relation_count": len(relations),
        "relations": relations,
    }


def _graph_entity_id(name: str) -> str:
    normalized = str(name or "unknown").strip().lower().replace(" ", "_")
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch in {"_", "-"})
    return normalized or "unknown"


def _relation_for_evidence(row: dict) -> str:
    evidence_type = row.get("evidence_type") or row.get("fact_type")
    if evidence_type == "recipe_component":
        return "has_component"
    if evidence_type == "table_column":
        return "has_column"
    if evidence_type == "data_value":
        return "has_value"
    if evidence_type == "sop_fact":
        return "uses_or_mentions"
    if evidence_type == "paper_claim":
        return "supports_topic"
    return "mentions"


def list_rag_source_schemas(
    doc_types: list[str] | None = None,
    source_file: str | None = None,
) -> list[dict]:
    params: list[Any] = [get_active_rag_generation_id()]
    filters = ["d.generation_id = ?", "d.status IN ('indexed', 'skipped')"]
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        filters.append(f"d.doc_type IN ({placeholders})")
        params.extend(doc_types)
    if source_file:
        filters.append("s.source_file = ?")
        params.append(source_file)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                s.*,
                d.doc_type,
                d.topic,
                d.version,
                d.year,
                d.language
            FROM rag_source_schemas s
            JOIN rag_documents d ON d.id = s.document_id
            {where_clause}
            ORDER BY s.source_file, s.sheet_name, s.column_index
            """,
            params,
        )
        rows = [dict(row) for row in cursor.fetchall()]
    for row in rows:
        try:
            row["sample_values"] = json.loads(row.pop("sample_values_json") or "[]")
        except Exception:
            row["sample_values"] = []
    return _filter_governed_rag_rows(rows)


def list_rag_recipe_components(
    doc_types: list[str] | None = None,
    source_file: str | None = None,
) -> list[dict]:
    params: list[Any] = [get_active_rag_generation_id()]
    filters = ["d.generation_id = ?", "d.status IN ('indexed', 'skipped')"]
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        filters.append(f"d.doc_type IN ({placeholders})")
        params.extend(doc_types)
    if source_file:
        filters.append("r.source_file = ?")
        params.append(source_file)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                r.*,
                d.doc_type,
                d.topic,
                d.version,
                d.year,
                d.language
            FROM rag_recipe_components r
            JOIN rag_documents d ON d.id = r.document_id
            {where_clause}
            ORDER BY r.source_file, r.table_index, r.id
            """,
            params,
        )
        return _filter_governed_rag_rows([dict(row) for row in cursor.fetchall()])


def list_rag_sop_facts(doc_types: list[str] | None = None) -> list[dict]:
    params: list[Any] = [get_active_rag_generation_id()]
    filters = ["d.generation_id = ?", "d.status IN ('indexed', 'skipped')"]
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        filters.append(f"d.doc_type IN ({placeholders})")
        params.extend(doc_types)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT f.*, d.doc_type, d.topic, d.version, d.year, d.language
            FROM rag_sop_facts f
            JOIN rag_documents d ON d.id = f.document_id
            {where_clause}
            ORDER BY f.source_file, f.page_number, f.id
            """,
            params,
        )
        return _filter_governed_rag_rows([dict(row) for row in cursor.fetchall()])


def list_rag_paper_facts(doc_types: list[str] | None = None) -> list[dict]:
    params: list[Any] = [get_active_rag_generation_id()]
    filters = ["d.generation_id = ?", "d.status IN ('indexed', 'skipped')"]
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        filters.append(f"d.doc_type IN ({placeholders})")
        params.extend(doc_types)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT f.*, d.doc_type, d.topic, d.version AS document_version, d.language
            FROM rag_paper_facts f
            JOIN rag_documents d ON d.id = f.document_id
            {where_clause}
            ORDER BY f.source_file, f.page_number, f.id
            """,
            params,
        )
        return _filter_governed_rag_rows([dict(row) for row in cursor.fetchall()])


def list_rag_experiment_data_values(doc_types: list[str] | None = None) -> list[dict]:
    params: list[Any] = [get_active_rag_generation_id()]
    filters = ["d.generation_id = ?", "d.status IN ('indexed', 'skipped')"]
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        filters.append(f"d.doc_type IN ({placeholders})")
        params.extend(doc_types)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT v.*, d.doc_type, d.topic, d.version, d.year, d.language
            FROM rag_experiment_data_values v
            JOIN rag_documents d ON d.id = v.document_id
            {where_clause}
            ORDER BY v.source_file, v.sheet_name, v.row_index, v.id
            """,
            params,
        )
        return _filter_governed_rag_rows([dict(row) for row in cursor.fetchall()])


def _filter_governed_rag_rows(rows: list[dict]) -> list[dict]:
    """Apply source lifecycle, review, and current-version policy to structured evidence."""
    if not rows:
        return rows
    sources = list_rag_knowledge_sources()
    now = datetime.now().astimezone()
    result = []
    for row in rows:
        candidates = [
            source for source in sources
            if (
                row.get("source_id") and source.get("source_id") == row.get("source_id")
            ) or (
                row.get("source_file") and str(source.get("file_name") or "").casefold() == str(row.get("source_file")).casefold()
            )
        ]
        if not candidates or any(_source_is_current(source, now) for source in candidates):
            result.append(row)
    return result


def _source_is_current(source: dict, instant: datetime) -> bool:
    if source.get("lifecycle_status", "active") != "active" or source.get("review_status", "approved") != "approved":
        return False
    start = _parse_rag_datetime(source.get("effective_from"))
    end = _parse_rag_datetime(source.get("effective_to"))
    return not ((start and instant < start) or (end and instant >= end))


def _parse_rag_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.astimezone()


def _rag_source_scope_sql(
    source_ids: list[str] | None,
    version_policy: str,
    as_of: str | None,
) -> tuple[str, list[Any]]:
    clauses = [
        "COALESCE(s.lifecycle_status, 'active') = 'active'",
        "COALESCE(s.review_status, 'approved') = 'approved'",
    ]
    params: list[Any] = []
    if source_ids:
        placeholders = ",".join("?" for _ in source_ids)
        clauses.append(f"s.source_id IN ({placeholders})")
        params.extend(source_ids)
    if version_policy in {"current", "as_of"}:
        instant = as_of if version_policy == "as_of" and as_of else datetime.now().astimezone().isoformat()
        clauses.extend([
            "(s.effective_from IS NULL OR s.effective_from <= ?)",
            "(s.effective_to IS NULL OR s.effective_to > ?)",
        ])
        params.extend([instant, instant])
    return " AND " + " AND ".join(clauses), params


def search_rag_chunks(
    query: str,
    top_k: int = 5,
    doc_types: list[str] | None = None,
    generation_id: str | None = None,
    source_ids: list[str] | None = None,
    version_policy: str = "current",
    as_of: str | None = None,
) -> list[dict]:
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []
    generation_id = generation_id or get_active_rag_generation_id()
    params: list[Any] = [fts_query, generation_id]
    doc_type_filter = ""
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        doc_type_filter = f" AND c.doc_type IN ({placeholders})"
        params.extend(doc_types)
    source_scope, source_params = _rag_source_scope_sql(source_ids, version_policy, as_of)
    params.extend(source_params)
    params.append(max(int(top_k or 5) * 3, 5))

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"""
                SELECT
                    c.*,
                    d.topic,
                    d.version,
                    d.year,
                    d.language,
                    bm25(rag_chunks_fts) AS score
                FROM rag_chunks_fts
                JOIN rag_chunks c ON c.chunk_id = rag_chunks_fts.chunk_id
                JOIN rag_documents d ON d.id = c.document_id
                JOIN rag_knowledge_sources s ON s.source_id = d.source_id
                WHERE rag_chunks_fts MATCH ? AND d.generation_id = ?
                  AND d.status IN ('indexed', 'skipped') {doc_type_filter} {source_scope}
                ORDER BY score
                LIMIT ?
                """,
                params,
            )
            rows = [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            rows = []

        if not rows:
            like_terms = [term for term in _query_tokens(query)[:12] if term.strip()]
            if not like_terms:
                return []
            like_clause = " OR ".join(
                [
                    "(c.content LIKE ? OR c.file_name LIKE ? OR c.title LIKE ? OR c.section LIKE ?)"
                    for _ in like_terms
                ]
            )
            params = [generation_id]
            for term in like_terms:
                like_term = f"%{term}%"
                params.extend([like_term, like_term, like_term, like_term])
            doc_type_filter = ""
            if doc_types:
                placeholders = ",".join("?" for _ in doc_types)
                doc_type_filter = f" AND c.doc_type IN ({placeholders})"
                params.extend(doc_types)
            params.extend(source_params)
            params.append(max(int(top_k or 5) * 3, 5))
            cursor.execute(
                f"""
                SELECT
                    c.*,
                    d.topic,
                    d.version,
                    d.year,
                    d.language,
                    0 AS score
                FROM rag_chunks c
                JOIN rag_documents d ON d.id = c.document_id
                JOIN rag_knowledge_sources s ON s.source_id = d.source_id
                WHERE d.generation_id = ? AND ({like_clause})
                  AND d.status IN ('indexed', 'skipped') {doc_type_filter} {source_scope}
                LIMIT ?
                """,
                params,
            )
            rows = [dict(row) for row in cursor.fetchall()]

    for row in rows:
        try:
            row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
        except Exception:
            row["metadata"] = {}
    return rows[: max(int(top_k or 5), 1)]


def list_rag_chunks_for_semantic(
    doc_types: list[str] | None = None,
    limit: int = 1000,
    generation_id: str | None = None,
    source_ids: list[str] | None = None,
    version_policy: str = "current",
    as_of: str | None = None,
) -> list[dict]:
    params: list[Any] = [generation_id or get_active_rag_generation_id()]
    doc_type_filter = ""
    if doc_types:
        placeholders = ",".join("?" for _ in doc_types)
        doc_type_filter = f"AND c.doc_type IN ({placeholders})"
        params.extend(doc_types)
    source_scope, source_params = _rag_source_scope_sql(source_ids, version_policy, as_of)
    params.extend(source_params)
    params.append(max(int(limit or 1000), 1))

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                c.*,
                d.topic,
                d.version,
                d.year,
                d.language
            FROM rag_chunks c
            JOIN rag_documents d ON d.id = c.document_id
            JOIN rag_knowledge_sources s ON s.source_id = d.source_id
            WHERE d.generation_id = ? AND d.status IN ('indexed', 'skipped')
            {doc_type_filter} {source_scope}
            ORDER BY d.doc_type, d.file_name, c.chunk_index
            LIMIT ?
            """,
            params,
        )
        rows = [dict(row) for row in cursor.fetchall()]

    for row in rows:
        try:
            row["metadata"] = json.loads(row.pop("metadata_json") or "{}")
        except Exception:
            row["metadata"] = {}
    return rows


def stamp_rag_document_generation(document_id: int, generation_id: str) -> None:
    """Stamp every derived row after ingestion; legacy adapters need no signature changes."""
    with sqlite3.connect(DB_PATH) as conn:
        for table in (
            "rag_chunks", "rag_chunk_embeddings", "rag_source_schemas",
            "rag_recipe_components", "rag_sop_facts", "rag_paper_facts",
            "rag_experiment_data_values", "rag_evidence_units",
            "rag_hierarchy_nodes",
        ):
            conn.execute(
                f"UPDATE {table} SET generation_id = ? WHERE document_id = ?",
                (generation_id, document_id),
            )
        conn.commit()


def get_rag_document_integrity(
    document_id: int,
    *,
    element_document_id: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    """Return derived-layer counts used to repair partial ingestions safely."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        document = conn.execute(
            "SELECT generation_id, chunk_count, index_schema_version FROM rag_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if not document:
            return {"exists": False}
        generation_id = str(document["generation_id"] or "legacy")
        chunk_count = int(conn.execute("SELECT COUNT(*) FROM rag_chunks WHERE document_id = ?", (document_id,)).fetchone()[0])
        embedding_sql = "SELECT COUNT(*) FROM rag_chunk_embeddings WHERE document_id = ?"
        embedding_params: list[Any] = [document_id]
        if embedding_model:
            embedding_sql += " AND embedding_model = ?"
            embedding_params.append(embedding_model)
        result = {
            "exists": True,
            "generation_id": generation_id,
            "index_schema_version": document["index_schema_version"],
            "chunks": chunk_count,
            "embeddings": int(conn.execute(embedding_sql, embedding_params).fetchone()[0]),
            "elements": int(conn.execute(
                "SELECT COUNT(*) FROM rag_document_elements WHERE generation_id = ? AND document_id = ?",
                (generation_id, element_document_id or str(document_id)),
            ).fetchone()[0]),
            "evidence": int(conn.execute("SELECT COUNT(*) FROM rag_evidence_units WHERE document_id = ?", (document_id,)).fetchone()[0]),
            "hierarchy_nodes": int(conn.execute("SELECT COUNT(*) FROM rag_hierarchy_nodes WHERE document_id = ?", (document_id,)).fetchone()[0]),
        }
        for key, table in (
            ("schemas", "rag_source_schemas"),
            ("recipes", "rag_recipe_components"),
            ("sop_facts", "rag_sop_facts"),
            ("paper_facts", "rag_paper_facts"),
            ("experiment_values", "rag_experiment_data_values"),
        ):
            result[key] = int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE document_id = ?", (document_id,)).fetchone()[0])
        return result


def get_rag_generation_health(generation_id: str | None = None) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        generation_id = generation_id or get_active_rag_generation_id(conn)
        generation = conn.execute(
            "SELECT * FROM rag_index_generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
        docs = conn.execute(
            "SELECT * FROM rag_documents WHERE generation_id = ? AND status IN ('indexed', 'skipped')",
            (generation_id,),
        ).fetchall()
        doc_ids = [int(row["id"]) for row in docs]
        counts = {
            "documents": len(doc_ids), "chunks": 0, "elements": 0,
            "evidence_units": 0, "embeddings": 0, "hierarchy_nodes": 0,
        }
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            for key, table in (
                ("chunks", "rag_chunks"),
                ("evidence_units", "rag_evidence_units"),
                ("embeddings", "rag_chunk_embeddings"),
                ("hierarchy_nodes", "rag_hierarchy_nodes"),
            ):
                counts[key] = int(conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE document_id IN ({placeholders})", doc_ids
                ).fetchone()[0])
            counts["elements"] = int(conn.execute(
                "SELECT COUNT(*) FROM rag_document_elements WHERE generation_id = ?", (generation_id,)
            ).fetchone()[0])
        canonical_paths = [
            _canonical_source_path(row["canonical_source_path"] or row["source_path"]).casefold()
            for row in docs
        ]
        duplicate_paths = len(canonical_paths) - len(set(canonical_paths))
        source_rows = conn.execute(
            "SELECT source_id, canonical_source_path, source_path, content_hash FROM rag_knowledge_sources WHERE lifecycle_status = 'active'"
        ).fetchall()
        source_paths = [
            _canonical_source_path(row["canonical_source_path"] or row["source_path"]).casefold()
            for row in source_rows
        ]
        duplicate_source_paths = len(source_paths) - len(set(source_paths))
        content_hash_counts: dict[str, int] = {}
        for row in source_rows:
            if row["content_hash"]:
                content_hash_counts[str(row["content_hash"])] = content_hash_counts.get(str(row["content_hash"]), 0) + 1
        duplicate_content_hashes = [key for key, count in content_hash_counts.items() if count > 1]
        forbidden = [
            row["canonical_source_path"] or row["source_path"]
            for row in docs
            if any(part in str(row["canonical_source_path"] or row["source_path"]).replace("\\", "/").casefold()
                   for part in ("/.tmp/", "/test_workspaces/", "/tests/"))
        ]
        chunk_count = counts["chunks"]
        embedding_required = bool(generation and generation["embedding_model"])
        coverage = {
            "element_document_coverage": _document_coverage(conn, "rag_document_elements", doc_ids, generation_id),
            "evidence_document_coverage": _document_coverage(conn, "rag_evidence_units", doc_ids, generation_id),
            "embedding_chunk_coverage": (counts["embeddings"] / chunk_count) if chunk_count else 0.0,
        }
        embedding_profiles = [dict(row) for row in conn.execute(
            "SELECT embedding_model, dimension, COUNT(*) AS count FROM rag_chunk_embeddings WHERE generation_id = ? GROUP BY embedding_model, dimension",
            (generation_id,),
        ).fetchall()]
        configured_dimension = int(generation["embedding_dimension"] or 0) if generation else 0
        embedding_profile_valid = (
            not embedding_required
            or (
                len(embedding_profiles) == 1
                and (not configured_dimension or int(embedding_profiles[0]["dimension"]) == configured_dimension)
                and str(embedding_profiles[0]["embedding_model"]) == str(generation["embedding_model"])
            )
        )
        activation_ready = bool(
            doc_ids
            and duplicate_paths == 0
            and duplicate_source_paths == 0
            and not forbidden
            and coverage["element_document_coverage"] >= 1.0
            and coverage["evidence_document_coverage"] >= 1.0
            and (not embedding_required or coverage["embedding_chunk_coverage"] >= 1.0)
            and embedding_profile_valid
        )
        return {
            "generation_id": generation_id,
            "counts": counts,
            "coverage": coverage,
            "duplicate_canonical_paths": duplicate_paths,
            "duplicate_source_paths": duplicate_source_paths,
            "duplicate_content_hashes": duplicate_content_hashes,
            "embedding_profiles": embedding_profiles,
            "embedding_profile_valid": embedding_profile_valid,
            "forbidden_sources": forbidden,
            "activation_ready": activation_ready,
        }


def _document_coverage(
    conn: sqlite3.Connection,
    table: str,
    document_ids: list[int],
    generation_id: str,
) -> float:
    if not document_ids:
        return 0.0
    if table == "rag_document_elements":
        # Element document ids are parser-stable strings, so coverage is generation count based.
        count = int(conn.execute(
            "SELECT COUNT(DISTINCT document_id) FROM rag_document_elements WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()[0])
    else:
        placeholders = ",".join("?" for _ in document_ids)
        count = int(conn.execute(
            f"SELECT COUNT(DISTINCT document_id) FROM {table} WHERE document_id IN ({placeholders})",
            document_ids,
        ).fetchone()[0])
    return min(count / len(document_ids), 1.0)


def get_rag_index_status() -> dict:
    active_generation_id = get_active_rag_generation_id()
    docs = list_rag_documents(generation_id=active_generation_id)
    counts: dict[str, int] = {}
    for doc in docs:
        counts[doc["status"]] = counts.get(doc["status"], 0) + 1
    schema_status = _get_rag_schema_status(docs)
    return {
        "active_generation_id": active_generation_id,
        "generations": list_rag_index_generations(),
        "health": get_rag_generation_health(active_generation_id),
        "document_count": len(docs),
        "status_counts": counts,
        "schema_status": schema_status,
        "documents": docs,
    }


def _get_rag_schema_status(docs: list[dict]) -> dict:
    experiment_data_docs = [
        doc
        for doc in docs
        if doc.get("doc_type") == "experiment_data"
        and str(doc.get("file_name") or "").lower().endswith((".csv", ".xlsx", ".xls"))
    ]
    if not experiment_data_docs:
        return {
            "schema_count": 0,
            "schema_document_count": 0,
            "documents_missing_schema": [],
            "warnings": [],
        }

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT document_id, COUNT(*) AS schema_count
            FROM rag_source_schemas
            GROUP BY document_id
            """
        )
        counts_by_document_id = {
            int(row["document_id"]): int(row["schema_count"]) for row in cursor.fetchall()
        }

    missing = [
        doc
        for doc in experiment_data_docs
        if counts_by_document_id.get(int(doc["id"]), 0) == 0
    ]
    schema_count = sum(counts_by_document_id.values())
    warnings = []
    if missing:
        warnings.append(
            "experiment_data documents exist but some have no table schema; "
            "run `python -m app.services.rag.ingestion.ingest --source data/raw --backfill-schema`."
        )
    return {
        "schema_count": schema_count,
        "schema_document_count": sum(1 for count in counts_by_document_id.values() if count > 0),
        "documents_missing_schema": [
            {
                "id": doc["id"],
                "file_name": doc["file_name"],
                "source_path": doc["source_path"],
                "status": doc["status"],
            }
            for doc in missing
        ],
        "warnings": warnings,
    }


def insert_rag_query_log(
    question: str,
    answer: str | None,
    citations: list[dict] | None,
    uncertainty: list[str] | None,
    blocked_reason: str | None = None,
) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO rag_query_logs
            (question, answer, citations_json, uncertainty_json, blocked_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                answer,
                json.dumps(citations or [], ensure_ascii=False),
                json.dumps(uncertainty or [], ensure_ascii=False),
                blocked_reason,
                local_time_string(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def _build_fts_query(query: str) -> str:
    tokens = _query_tokens(query)
    if not tokens:
        text = query.strip()
        return f'"{text}"' if text else ""
    return " OR ".join(f'"{token}"' for token in tokens[:12])


def _query_tokens(query: str) -> list[str]:
    from app.services.rag.query_normalizer import sparse_query_tokens

    return sparse_query_tokens(query, limit=24)

def _space_mixed_text(text: str) -> str:
    spaced = []
    previous = ""
    for char in text:
        if previous and (_is_ascii_alnum(previous) != _is_ascii_alnum(char)):
            spaced.append(" ")
        spaced.append(char)
        previous = char
    return "".join(spaced)


def _is_ascii_alnum(char: str) -> bool:
    return char.isascii() and char.isalnum()
