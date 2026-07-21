import argparse
from collections import Counter

from app.core.database import init_db
from app.services.rag.ingestion.index_store import ingest_file
from app.services.rag.ingestion.metadata import iter_supported_files


def ingest_sources(source: str, rebuild: bool = False, dry_run: bool = False) -> dict:
    init_db()
    files = iter_supported_files(source)
    results = [ingest_file(path, rebuild=rebuild, dry_run=dry_run) for path in files]
    counts = Counter(item["status"] for item in results)
    return {
        "source": source,
        "total": len(results),
        "counts": dict(counts),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local lab knowledge into the RAG index.")
    parser.add_argument("--source", default="data/raw", help="File or directory to ingest.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index even when file hash is unchanged.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed without writing DB rows.")
    parser.add_argument(
        "--backfill-schema",
        action="store_true",
        help="Backfill structured table schema for already-ingested CSV/XLSX experiment data.",
    )
    args = parser.parse_args()

    if args.backfill_schema:
        from app.services.rag.evidence.backfill_schema import backfill_table_schemas

        summary = backfill_table_schemas(args.source, dry_run=args.dry_run)
    else:
        summary = ingest_sources(args.source, rebuild=args.rebuild, dry_run=args.dry_run)
    print(f"source: {summary['source']}")
    print(f"total: {summary['total']}")
    for status, count in sorted(summary["counts"].items()):
        print(f"{status}: {count}")
    for item in summary["results"]:
        detail = f" chunks={item.get('chunk_count', 0)}"
        if "schema_count" in item:
            detail += f" schemas={item.get('schema_count', 0)}"
        if item.get("reason"):
            detail += f" reason={item['reason']}"
        if item.get("error"):
            detail += f" error={item['error']}"
        print(f"- {item['status']}: {item['source_path']}{detail}")


if __name__ == "__main__":
    main()
