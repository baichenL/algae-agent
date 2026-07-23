from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.core.database import (
    get_active_rag_generation_id,
    get_rag_document_by_path,
    get_rag_document_integrity,
    mark_rag_document_failed,
    mark_rag_document_indexed,
    mark_rag_document_skipped,
    replace_rag_chunk_embeddings,
    replace_rag_chunks,
    replace_rag_document_elements,
    replace_rag_evidence_units,
    replace_rag_hierarchy_nodes,
    replace_rag_experiment_data_values,
    replace_rag_paper_facts,
    replace_rag_recipe_components,
    replace_rag_sop_facts,
    replace_rag_source_schemas,
    stamp_rag_document_generation,
    upsert_rag_document,
    upsert_rag_knowledge_source,
)
from app.core.db.rag import RAG_INDEX_SCHEMA_VERSION
from app.services.observability.error_events import record_error_event
from app.services.rag.embedding_service import (
    active_vector_backend,
    build_chunk_embedding_records,
    embedding_runtime_status,
    get_embedding_config,
)
from app.services.rag.evidence.adapters.chunk_adapter import chunks_to_evidence_units
from app.services.rag.evidence.adapters.experiment_data_adapter import (
    experiment_value_rows_to_evidence_units,
    extract_experiment_data_value_rows,
)
from app.services.rag.evidence.adapters.paper_adapter import extract_paper_fact_rows, paper_rows_to_evidence_units
from app.services.rag.evidence.adapters.recipe_adapter import extract_recipe_component_rows, recipe_rows_to_evidence_units
from app.services.rag.evidence.adapters.sop_adapter import extract_sop_fact_rows, sop_rows_to_evidence_units
from app.services.rag.evidence.adapters.table_schema_adapter import extract_table_schema_rows, schema_rows_to_evidence_units
from app.services.rag.ingestion.chunker import build_chunks
from app.services.rag.ingestion.loaders import load_source_units
from app.services.rag.ingestion.metadata import compute_file_hash, parse_source_metadata
from app.services.rag.ingestion.parsers import normalized_document_to_loaded_units, parse_document
from app.services.rag.security import scan_knowledge_text
from app.services.rag.hierarchy import build_hierarchy_nodes


def ingest_file(
    path: str | Path,
    rebuild: bool = False,
    dry_run: bool = False,
    metadata_overrides: dict | None = None,
    generation_id: str | None = None,
) -> dict:
    source = Path(path).resolve()
    overrides = dict(metadata_overrides or {})
    generation_id = generation_id or get_active_rag_generation_id()
    metadata = {**parse_source_metadata(source), **overrides}
    metadata.update({"source_path": str(source), "file_name": metadata.get("file_name") or source.name, "extension": source.suffix.lower()})
    content_hash = compute_file_hash(source)

    if dry_run:
        return {"source_path": str(source), "generation_id": generation_id, "status": "would_index", "chunk_count": 0}

    source_id = upsert_rag_knowledge_source(
        source_path=str(source),
        file_name=metadata["file_name"],
        doc_type=metadata["doc_type"],
        version=metadata.get("version"),
        year=metadata.get("year"),
        language=metadata.get("language"),
        ingestion_status="indexing",
        content_hash=content_hash,
        metadata={"topic": metadata.get("topic"), "extension": metadata.get("extension")},
        chunk_count=0,
        asset_key=overrides.get("asset_key"),
        effective_from=overrides.get("effective_from"),
        effective_to=overrides.get("effective_to"),
        supersedes_source_id=overrides.get("supersedes_source_id"),
    )
    metadata["source_id"] = source_id
    existing = get_rag_document_by_path(metadata["source_path"], generation_id=generation_id)
    document_id = upsert_rag_document(
        source_path=str(source),
        file_name=metadata["file_name"],
        doc_type=metadata["doc_type"],
        topic=metadata.get("topic"),
        version=metadata.get("version"),
        year=metadata.get("year"),
        language=metadata.get("language"),
        content_hash=content_hash,
        status="indexing",
        generation_id=generation_id,
        source_id=source_id,
    )

    try:
        _ensure_loader_can_read(source, metadata)
        normalized_document = parse_document(source, {**metadata, "content_hash": content_hash})
        units = normalized_document_to_loaded_units(normalized_document)
        scan = scan_knowledge_text("\n".join(unit.content for unit in units))
        quality_report = (normalized_document.metadata or {}).get("quality_report") or {}
        parser_degraded = quality_report.get("status") == "degraded"
        review_flags = list(scan.flags)
        if parser_degraded:
            review_flags.extend(
                f"parser_degraded:{flag}" for flag in (quality_report.get("degraded_flags") or ["unspecified"])
            )
        element_document_id = f"{generation_id}:{normalized_document.document_id}"
        elements = _namespaced_elements(normalized_document, generation_id, element_document_id)
        chunks = build_chunks(units, metadata, generation_id=generation_id)
        hierarchy_nodes = (
            build_hierarchy_nodes(chunks, generation_id=generation_id, document_id=document_id)
            if os.getenv("RAG_HIERARCHY_INDEX_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
            else []
        )

        schema_rows = extract_table_schema_rows(source, metadata, document_id)
        recipe_rows = extract_recipe_component_rows(source, metadata, document_id)
        sop_rows = extract_sop_fact_rows(source, metadata, document_id)
        paper_rows = extract_paper_fact_rows(source, metadata, document_id)
        data_rows = extract_experiment_data_value_rows(source, metadata, document_id)
        evidence_units = _namespaced_evidence(
            [
                *chunks_to_evidence_units(chunks),
                *schema_rows_to_evidence_units(schema_rows),
                *recipe_rows_to_evidence_units(recipe_rows),
                *sop_rows_to_evidence_units(sop_rows),
                *paper_rows_to_evidence_units(paper_rows),
                *experiment_value_rows_to_evidence_units(data_rows),
            ],
            generation_id=generation_id,
            source_id=source_id,
            content_hash=content_hash,
            document_version=normalized_document.document_version,
        )
        integrity = get_rag_document_integrity(
            document_id,
            element_document_id=element_document_id,
            embedding_model=get_embedding_config().model,
        )
        changed = not existing or existing.get("content_hash") != content_hash
        schema_changed = integrity.get("index_schema_version") != RAG_INDEX_SCHEMA_VERSION
        force = bool(rebuild or changed or schema_changed)
        repaired_layers: list[str] = []

        if force or integrity.get("elements") != len(elements):
            replace_rag_document_elements(element_document_id, elements)
            repaired_layers.append("document_elements")
        chunks_replaced = force or integrity.get("chunks") != len(chunks)
        if chunks_replaced:
            replace_rag_chunks(document_id, chunks)
            repaired_layers.append("chunks")
        if force or integrity.get("hierarchy_nodes") != len(hierarchy_nodes):
            replace_rag_hierarchy_nodes(document_id, generation_id, hierarchy_nodes)
            repaired_layers.append("hierarchy_nodes")

        _replace_if_missing(force, integrity, "schemas", schema_rows, replace_rag_source_schemas, document_id, repaired_layers)
        _replace_if_missing(force, integrity, "recipes", recipe_rows, replace_rag_recipe_components, document_id, repaired_layers)
        _replace_if_missing(force, integrity, "sop_facts", sop_rows, replace_rag_sop_facts, document_id, repaired_layers)
        _replace_if_missing(force, integrity, "paper_facts", paper_rows, replace_rag_paper_facts, document_id, repaired_layers)
        _replace_if_missing(force, integrity, "experiment_values", data_rows, replace_rag_experiment_data_values, document_id, repaired_layers)
        if force or integrity.get("evidence") != len(evidence_units):
            replace_rag_evidence_units(document_id, source_id, evidence_units)
            repaired_layers.append("evidence_units")

        embedding_status = embedding_runtime_status()
        embedding_required = embedding_status["status"] not in {"disabled", "disabled_missing_api_key"}
        if embedding_required and (chunks_replaced or integrity.get("embeddings") != len(chunks)):
            embedding_status = _index_chunk_embeddings(document_id, chunks)
            repaired_layers.append("embeddings")
        stamp_rag_document_generation(document_id, generation_id)

        review_required = bool(not scan.safe or parser_degraded)
        final_status = "review_required" if review_required else "indexed"
        upsert_rag_document(
            source_path=str(source), file_name=metadata["file_name"], doc_type=metadata["doc_type"],
            topic=metadata.get("topic"), version=metadata.get("version"), year=metadata.get("year"),
            language=metadata.get("language"), content_hash=content_hash, status=final_status,
            error_message="review_flags:" + ",".join(review_flags) if review_flags else None,
            generation_id=generation_id, source_id=source_id,
        )
        if not review_required:
            if repaired_layers:
                mark_rag_document_indexed(document_id, len(chunks))
            else:
                mark_rag_document_skipped(document_id)
        upsert_rag_knowledge_source(
            source_path=str(source), file_name=metadata["file_name"], doc_type=metadata["doc_type"],
            version=metadata.get("version"), year=metadata.get("year"), language=metadata.get("language"),
            ingestion_status=final_status, content_hash=content_hash,
            metadata={"topic": metadata.get("topic"), "extension": metadata.get("extension")},
            chunk_count=len(chunks), asset_key=overrides.get("asset_key"),
            effective_from=overrides.get("effective_from"), effective_to=overrides.get("effective_to"),
            supersedes_source_id=overrides.get("supersedes_source_id"), security_flags=review_flags,
            review_status="approved" if not review_required else "review_required",
        )
        return {
            "source_path": str(source), "source_id": source_id, "generation_id": generation_id,
            "status": final_status if review_required else "indexed" if repaired_layers else "skipped",
            "chunk_count": len(chunks), "repaired_layers": repaired_layers,
            "embedding_status": embedding_status, "security_flags": scan.flags,
            "review_flags": review_flags,
        }
    except Exception as exc:
        mark_rag_document_failed(document_id, str(exc))
        upsert_rag_knowledge_source(
            source_path=str(source), file_name=metadata["file_name"], doc_type=metadata["doc_type"],
            version=metadata.get("version"), year=metadata.get("year"), language=metadata.get("language"),
            ingestion_status="failed", content_hash=content_hash,
            metadata={"topic": metadata.get("topic"), "error": str(exc)}, chunk_count=0, last_error=str(exc),
            asset_key=overrides.get("asset_key"), effective_from=overrides.get("effective_from"),
            effective_to=overrides.get("effective_to"), supersedes_source_id=overrides.get("supersedes_source_id"),
        )
        return {"source_path": str(source), "generation_id": generation_id, "status": "failed", "chunk_count": 0, "error": str(exc)}


def _replace_if_missing(force, integrity, key, rows, function, document_id, repaired_layers) -> None:
    if force or integrity.get(key) != len(rows):
        function(document_id, rows)
        repaired_layers.append(key)


def _index_chunk_embeddings(document_id: int, chunks: list[dict]) -> dict:
    status = embedding_runtime_status()
    if _strict_mode() and status.get("degraded"):
        raise RuntimeError(f"Strict production mode rejects degraded vector backend: {status.get('degraded_reason')}")
    if _strict_mode() and status["status"] in {"disabled", "disabled_missing_api_key"} and _dense_retrieval_enabled():
        raise RuntimeError(f"Strict production mode requires embeddings for dense retrieval: {status['status']}")
    if status["status"] in {"disabled", "disabled_missing_api_key"}:
        return status
    try:
        records = build_chunk_embedding_records(chunks)
        if records:
            replace_rag_chunk_embeddings(document_id, records, get_embedding_config().model, active_vector_backend())
        return {**status, "indexed_count": len(records)}
    except Exception as exc:
        record_error_event(
            layer="rag", component="rag_ingestion", operation="index_chunk_embeddings", severity="warning",
            error_type=type(exc).__name__, error_message=str(exc),
            metadata={"document_id": document_id, "chunk_count": len(chunks)},
        )
        if _strict_mode():
            raise
        return {**status, "status": "embedding_failed", "error": str(exc), "indexed_count": 0}


def _namespaced_elements(document, generation_id: str, document_id: str):
    id_map = {element.element_id: f"{generation_id}:{element.element_id}" for element in document.elements}
    return [
        element.model_copy(
            update={
                "element_id": id_map[element.element_id], "document_id": document_id,
                "parent_id": id_map.get(element.parent_id, element.parent_id),
                "previous_id": id_map.get(element.previous_id, element.previous_id),
                "next_id": id_map.get(element.next_id, element.next_id),
                "metadata": {
                    **(element.metadata or {}), "generation_id": generation_id,
                    "document_version": document.document_version, "parser_name": document.parser_name,
                    "parser_version": document.parser_version,
                },
            }
        )
        for element in document.elements
    ]


def _namespaced_evidence(units, *, generation_id: str, source_id: str, content_hash: str, document_version: str):
    result = []
    for unit in units:
        row = unit.model_dump() if hasattr(unit, "model_dump") else dict(unit)
        evidence_id = f"{generation_id}:{row['evidence_id']}"
        locator = row.get("source_locator") or row.get("location") or {}
        citation = {
            **(row.get("citation") or {}), "knowledge_source_id": source_id,
            "generation_id": generation_id, "evidence_id": evidence_id,
            "document_version": row.get("document_version") or document_version,
            "content_hash": row.get("content_hash") or content_hash, "source_locator": locator,
        }
        result.append(
            unit.model_copy(
                update={
                    "evidence_id": evidence_id, "source_id": source_id,
                    "document_version": row.get("document_version") or document_version,
                    "citation": citation,
                    "metadata": {**(row.get("metadata") or {}), "generation_id": generation_id},
                }
            )
        )
    return result


def _strict_mode() -> bool:
    return os.getenv("RAG_STRICT_PRODUCTION", "false").strip().lower() in {"1", "true", "yes", "on"}


def _dense_retrieval_enabled() -> bool:
    return os.getenv("RAG_DENSE_RETRIEVAL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_loader_can_read(source: Path, metadata: dict) -> None:
    load_source_units(source, metadata)
