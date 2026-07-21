from pathlib import Path

from app.core.database import (
    get_rag_document_by_path,
    init_db,
    list_rag_documents,
    replace_rag_source_schemas,
)
from app.services.rag.ingestion.metadata import iter_supported_files, parse_source_metadata
from app.services.rag.evidence.adapters.table_schema_adapter import extract_table_schema_rows


TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def backfill_table_schemas(source: str | Path = "data/raw", dry_run: bool = False) -> dict:
    init_db()
    files = [
        path
        for path in iter_supported_files(source)
        if path.suffix.lower() in TABLE_EXTENSIONS
    ]
    documents = list_rag_documents()
    results = [_backfill_file(path, documents, dry_run=dry_run) for path in files]
    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "source": str(source),
        "total": len(results),
        "counts": counts,
        "results": results,
    }


def _backfill_file(path: Path, documents: list[dict], dry_run: bool = False) -> dict:
    document = _find_document_for_path(path, documents)
    if not document:
        return {
            "source_path": str(path),
            "status": "missing_document",
            "schema_count": 0,
            "reason": "file_has_not_been_ingested",
        }

    metadata = parse_source_metadata(path)
    if metadata.get("doc_type") != "experiment_data":
        return {
            "source_path": str(path),
            "status": "skipped",
            "schema_count": 0,
            "reason": "not_experiment_data",
        }

    try:
        schema_rows = extract_table_schema_rows(path, metadata, int(document["id"]))
    except Exception as exc:
        return {
            "source_path": str(path),
            "status": "failed",
            "schema_count": 0,
            "error": str(exc),
        }

    if dry_run:
        return {
            "source_path": str(path),
            "status": "would_backfill",
            "schema_count": len(schema_rows),
        }

    replace_rag_source_schemas(int(document["id"]), schema_rows)
    return {
        "source_path": str(path),
        "status": "backfilled",
        "schema_count": len(schema_rows),
        "document_id": int(document["id"]),
    }


def _find_document_for_path(path: Path, documents: list[dict]) -> dict | None:
    exact = get_rag_document_by_path(str(path))
    if exact:
        return exact

    try:
        resolved_path = path.resolve()
    except OSError:
        resolved_path = path

    resolved_matches = []
    for document in documents:
        source_path = document.get("source_path")
        if not source_path:
            continue
        try:
            if Path(source_path).resolve() == resolved_path:
                resolved_matches.append(document)
        except OSError:
            continue
    if len(resolved_matches) == 1:
        return resolved_matches[0]

    name_matches = [document for document in documents if document.get("file_name") == path.name]
    if len(name_matches) == 1:
        return name_matches[0]
    return None
