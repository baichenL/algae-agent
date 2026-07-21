from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

from app.core.db import rag as rag_db
from app.services.rag.ingestion.index_store import ingest_file
from app.services.rag.ingestion.metadata import SUPPORTED_EXTENSIONS


KNOWLEDGE_UPLOAD_ROOT = Path(os.getenv("KNOWLEDGE_UPLOAD_ROOT", "data/raw/uploads"))
MAX_KNOWLEDGE_UPLOAD_BYTES = int(os.getenv("MAX_KNOWLEDGE_UPLOAD_BYTES", str(50 * 1024 * 1024)))
KNOWLEDGE_DOC_TYPES = {
    "media_recipe", "manual", "paper", "experiment_data", "device_document", "document",
}


def _safe_name(filename: str) -> str:
    base = Path(filename or "document").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._") or "document"
    return f"{stem[:80]}{Path(base).suffix.lower()}"


def save_uploaded_source(
    *,
    content: bytes,
    filename: str,
    doc_type: str,
    owner: str,
    title: str | None = None,
    version: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    if doc_type not in KNOWLEDGE_DOC_TYPES:
        raise ValueError("invalid_doc_type")
    if not content:
        raise ValueError("empty_file")
    if len(content) > MAX_KNOWLEDGE_UPLOAD_BYTES:
        raise ValueError("knowledge_file_too_large")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("unsupported_extension")
    target_dir = (KNOWLEDGE_UPLOAD_ROOT / doc_type).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{uuid.uuid4().hex}__{_safe_name(filename)}"
    target.write_bytes(content)
    source_id = rag_db.upsert_rag_knowledge_source(
        source_path=str(target),
        file_name=Path(filename).name,
        doc_type=doc_type,
        version=version,
        language=language,
        owner=owner,
        ingestion_status="pending",
        metadata={"topic": title or Path(filename).stem, "extension": suffix, "original_filename": Path(filename).name},
        chunk_count=0,
    )
    return rag_db.get_rag_knowledge_source(source_id)


def index_source(source_id: str, *, rebuild: bool = False) -> dict[str, Any]:
    source = rag_db.get_rag_knowledge_source(source_id)
    if not source:
        return {"status": "error", "reason": "knowledge_source_not_found"}
    metadata = source.get("metadata") or {}
    return ingest_file(
        source["source_path"],
        rebuild=rebuild,
        metadata_overrides={
            "doc_type": source["doc_type"],
            "file_name": source["file_name"],
            "topic": metadata.get("topic") or Path(source["file_name"]).stem,
            "version": source.get("version"),
            "language": source.get("language"),
        },
    )


def list_sources(*, include_archived: bool = True) -> dict[str, Any]:
    sources = rag_db.list_rag_knowledge_sources()
    if not include_archived:
        sources = [item for item in sources if item.get("lifecycle_status", "active") == "active"]
    counts = {doc_type: 0 for doc_type in sorted(KNOWLEDGE_DOC_TYPES)}
    for item in sources:
        if item.get("lifecycle_status", "active") == "active":
            counts[item.get("doc_type") or "document"] = counts.get(item.get("doc_type") or "document", 0) + 1
    return {"sources": sources, "category_counts": counts}
