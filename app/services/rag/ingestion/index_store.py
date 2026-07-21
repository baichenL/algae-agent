from pathlib import Path
import os

from app.core.database import (
    consolidate_rag_document_path,
    get_rag_document_by_path,
    mark_rag_document_failed,
    mark_rag_document_indexed,
    mark_rag_document_skipped,
    replace_rag_chunk_embeddings,
    replace_rag_chunks,
    replace_rag_document_elements,
    replace_rag_evidence_units,
    replace_rag_experiment_data_values,
    replace_rag_paper_facts,
    replace_rag_recipe_components,
    replace_rag_sop_facts,
    replace_rag_source_schemas,
    upsert_rag_knowledge_source,
    upsert_rag_document,
)
from app.services.rag.ingestion.chunker import build_chunks
from app.services.rag.ingestion.loaders import load_source_units
from app.services.rag.ingestion.metadata import compute_file_hash, parse_source_metadata
from app.services.rag.ingestion.parsers import normalized_document_to_loaded_units, parse_document
from app.services.rag.evidence.adapters.experiment_data_adapter import extract_experiment_data_value_rows
from app.services.rag.evidence.adapters.experiment_data_adapter import experiment_value_rows_to_evidence_units
from app.services.rag.evidence.adapters.paper_adapter import extract_paper_fact_rows
from app.services.rag.evidence.adapters.paper_adapter import paper_rows_to_evidence_units
from app.services.rag.evidence.adapters.recipe_adapter import extract_recipe_component_rows
from app.services.rag.evidence.adapters.recipe_adapter import recipe_rows_to_evidence_units
from app.services.rag.evidence.adapters.sop_adapter import extract_sop_fact_rows
from app.services.rag.evidence.adapters.sop_adapter import sop_rows_to_evidence_units
from app.services.rag.evidence.adapters.table_schema_adapter import extract_table_schema_rows
from app.services.rag.evidence.adapters.table_schema_adapter import schema_rows_to_evidence_units
from app.services.rag.evidence.adapters.chunk_adapter import chunks_to_evidence_units
from app.services.rag.embedding_service import (
    active_vector_backend,
    build_chunk_embedding_records,
    embedding_runtime_status,
    get_embedding_config,
)
from app.services.observability.error_events import record_error_event


def ingest_file(
    path: str | Path,
    rebuild: bool = False,
    dry_run: bool = False,
    metadata_overrides: dict | None = None,
) -> dict:
    source = Path(path).resolve()
    metadata = {**parse_source_metadata(source), **(metadata_overrides or {})}
    metadata["source_path"] = str(source)
    metadata["file_name"] = metadata.get("file_name") or source.name
    metadata["extension"] = source.suffix.lower()
    content_hash = compute_file_hash(source)
    existing = None if dry_run else consolidate_rag_document_path(metadata["source_path"])
    if existing is None:
        existing = get_rag_document_by_path(metadata["source_path"])

    if (
        existing
        and existing.get("content_hash") == content_hash
        and not rebuild
        and not existing.get("_duplicates_merged")
    ):
        if not dry_run:
            mark_rag_document_skipped(existing["id"])
            source_id = upsert_rag_knowledge_source(
                source_path=str(source),
                file_name=metadata["file_name"],
                doc_type=metadata["doc_type"],
                version=metadata.get("version"),
                year=metadata.get("year"),
                language=metadata.get("language"),
                ingestion_status="skipped",
                content_hash=content_hash,
                metadata={"topic": metadata.get("topic"), "extension": metadata.get("extension")},
                chunk_count=int(existing.get("chunk_count") or 0),
            )
            _ensure_loader_can_read(source, metadata)
            parse_metadata = {**metadata, "content_hash": content_hash}
            normalized_document = parse_document(source, parse_metadata)
            replace_rag_document_elements(normalized_document.document_id, _elements_with_document_metadata(normalized_document))
            units = normalized_document_to_loaded_units(normalized_document)
            chunks = build_chunks(units, metadata)
            schema_rows = extract_table_schema_rows(source, metadata, int(existing["id"]))
            if schema_rows:
                replace_rag_source_schemas(int(existing["id"]), schema_rows)
            recipe_rows = extract_recipe_component_rows(source, metadata, int(existing["id"]))
            if recipe_rows:
                replace_rag_recipe_components(int(existing["id"]), recipe_rows)
            sop_rows = extract_sop_fact_rows(source, metadata, int(existing["id"]))
            if sop_rows:
                replace_rag_sop_facts(int(existing["id"]), sop_rows)
            paper_rows = extract_paper_fact_rows(source, metadata, int(existing["id"]))
            if paper_rows:
                replace_rag_paper_facts(int(existing["id"]), paper_rows)
            data_rows = extract_experiment_data_value_rows(source, metadata, int(existing["id"]))
            if data_rows:
                replace_rag_experiment_data_values(int(existing["id"]), data_rows)
            replace_rag_evidence_units(
                int(existing["id"]),
                source_id,
                [
                    *chunks_to_evidence_units(chunks),
                    *schema_rows_to_evidence_units(schema_rows),
                    *recipe_rows_to_evidence_units(recipe_rows),
                    *sop_rows_to_evidence_units(sop_rows),
                    *paper_rows_to_evidence_units(paper_rows),
                    *experiment_value_rows_to_evidence_units(data_rows),
                ],
            )
        return {
            "source_path": str(source),
            "status": "skipped",
            "chunk_count": existing.get("chunk_count", 0),
            "reason": "unchanged",
        }

    if dry_run:
        return {
            "source_path": str(source),
            "status": "would_index",
            "chunk_count": 0,
            "reason": "new_or_changed",
        }

    document_id = upsert_rag_document(
        source_path=str(source),
        file_name=metadata["file_name"],
        doc_type=metadata["doc_type"],
        topic=metadata.get("topic"),
        version=metadata.get("version"),
        year=metadata.get("year"),
        language=metadata.get("language"),
        content_hash=content_hash,
        status="pending",
    )
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
    )

    try:
        _ensure_loader_can_read(source, metadata)
        parse_metadata = {**metadata, "content_hash": content_hash}
        normalized_document = parse_document(source, parse_metadata)
        replace_rag_document_elements(normalized_document.document_id, _elements_with_document_metadata(normalized_document))
        units = normalized_document_to_loaded_units(normalized_document)
        chunks = build_chunks(units, metadata)
        replace_rag_chunks(document_id, chunks)
        embedding_status = _index_chunk_embeddings(document_id, chunks)
        schema_rows = extract_table_schema_rows(source, metadata, document_id)
        replace_rag_source_schemas(document_id, schema_rows)
        recipe_rows = extract_recipe_component_rows(source, metadata, document_id)
        replace_rag_recipe_components(document_id, recipe_rows)
        sop_rows = extract_sop_fact_rows(source, metadata, document_id)
        replace_rag_sop_facts(document_id, sop_rows)
        paper_rows = extract_paper_fact_rows(source, metadata, document_id)
        replace_rag_paper_facts(document_id, paper_rows)
        data_rows = extract_experiment_data_value_rows(source, metadata, document_id)
        replace_rag_experiment_data_values(document_id, data_rows)
        evidence_units = [
            *chunks_to_evidence_units(chunks),
            *schema_rows_to_evidence_units(schema_rows),
            *recipe_rows_to_evidence_units(recipe_rows),
            *sop_rows_to_evidence_units(sop_rows),
            *paper_rows_to_evidence_units(paper_rows),
            *experiment_value_rows_to_evidence_units(data_rows),
        ]
        replace_rag_evidence_units(document_id, source_id, evidence_units)
        upsert_rag_knowledge_source(
            source_path=str(source),
            file_name=metadata["file_name"],
            doc_type=metadata["doc_type"],
            version=metadata.get("version"),
            year=metadata.get("year"),
            language=metadata.get("language"),
            ingestion_status="indexed",
            content_hash=content_hash,
            metadata={"topic": metadata.get("topic"), "extension": metadata.get("extension")},
            chunk_count=len(chunks),
            last_error=None,
        )
        mark_rag_document_indexed(document_id, len(chunks))
        return {
            "source_path": str(source),
            "status": "indexed",
            "chunk_count": len(chunks),
            "embedding_status": embedding_status,
        }
    except Exception as exc:
        upsert_rag_knowledge_source(
            source_path=str(source),
            file_name=metadata["file_name"],
            doc_type=metadata["doc_type"],
            version=metadata.get("version"),
            year=metadata.get("year"),
            language=metadata.get("language"),
            ingestion_status="failed",
            content_hash=content_hash,
            metadata={"topic": metadata.get("topic"), "error": str(exc)},
            chunk_count=0,
            last_error=str(exc),
        )
        mark_rag_document_failed(document_id, str(exc))
        return {
            "source_path": str(source),
            "status": "failed",
            "chunk_count": 0,
            "error": str(exc),
        }


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
        if not records:
            return {**status, "indexed_count": 0}
        replace_rag_chunk_embeddings(
            document_id=document_id,
            embeddings=records,
            embedding_model=get_embedding_config().model,
            vector_backend=active_vector_backend(),
        )
        return {**status, "indexed_count": len(records)}
    except Exception as exc:
        record_error_event(
            layer="rag",
            component="rag_ingestion",
            operation="index_chunk_embeddings",
            severity="warning",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"document_id": document_id, "chunk_count": len(chunks)},
        )
        return {**status, "status": "embedding_failed", "error": str(exc)}


def _strict_mode() -> bool:
    return os.getenv("RAG_STRICT_PRODUCTION", "false").strip().lower() in {"1", "true", "yes", "on"}


def _dense_retrieval_enabled() -> bool:
    return os.getenv("RAG_DENSE_RETRIEVAL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _elements_with_document_metadata(normalized_document):
    return [
        element.model_copy(
            update={
                "metadata": {
                    **(element.metadata or {}),
                    "document_version": normalized_document.document_version,
                    "parser_name": normalized_document.parser_name,
                    "parser_version": normalized_document.parser_version,
                }
            }
        )
        for element in normalized_document.elements
    ]


def _ensure_loader_can_read(source: Path, metadata: dict) -> None:
    # Preserve the original ingestion failure boundary: tests and operators can
    # still treat source loader failures as indexing failures even though Phase 2
    # routes successful parsing through NormalizedDocument.
    load_source_units(source, metadata)
