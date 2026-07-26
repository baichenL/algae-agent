from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

from app.core.workspace_storage_migration import _table_digest, _table_names


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("storage_mode") != "split":
        raise ValueError("manifest_storage_mode_is_not_split")

    legacy = _resolve_manifest_path(str(manifest["legacy_db_path"]), manifest_path)
    control = _resolve_manifest_path(str(manifest["control_db_path"]), manifest_path)
    domain = _resolve_manifest_path(str(manifest["domain_db_path"]), manifest_path)
    for label, path in (
        ("legacy", legacy),
        ("control", control),
        ("domain", domain),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}_database_missing:{path}")

    verification = manifest.get("verification") or {}
    failures: list[str] = []
    checked_tables = 0
    for group, split_path in (("control", control), ("domain", domain)):
        expected_tables = verification.get(group) or {}
        actual_tables = set(_table_names(split_path))
        declared_tables = set(expected_tables)
        missing = sorted(declared_tables - actual_tables)
        unexpected = sorted(actual_tables - declared_tables)
        if missing:
            failures.append(f"{group}_tables_missing:{','.join(missing)}")
        if unexpected:
            failures.append(f"{group}_tables_unexpected:{','.join(unexpected)}")
        for table, recorded in expected_tables.items():
            checked_tables += 1
            legacy_digest = _table_digest(legacy, table)
            split_digest = _table_digest(split_path, table)
            if legacy_digest != recorded.get("legacy"):
                failures.append(f"{group}:{table}:legacy_digest_changed")
            if split_digest != recorded.get("split"):
                failures.append(f"{group}:{table}:split_digest_changed")
            if legacy_digest != split_digest:
                failures.append(f"{group}:{table}:legacy_split_mismatch")

    domain_uri = f"file:{domain.as_posix()}?mode=ro"
    with closing(sqlite3.connect(domain_uri, uri=True)) as conn:
        conn.execute("PRAGMA query_only = ON")
        query_only = int(conn.execute("PRAGMA query_only").fetchone()[0])
        conn.execute("SELECT 1").fetchone()
    if query_only != 1:
        failures.append("domain_readonly_probe_failed")

    return {
        "status": "passed" if not failures else "failed",
        "manifest": str(manifest_path),
        "legacy_db": str(legacy),
        "control_db": str(control),
        "domain_db": str(domain),
        "checked_tables": checked_tables,
        "domain_readonly_probe": query_only == 1,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only verification of a split workspace storage manifest."
    )
    parser.add_argument(
        "--manifest",
        default="data/workspace_storage_manifest.json",
    )
    args = parser.parse_args()
    try:
        result = verify_manifest(Path(args.manifest))
    except Exception as exc:
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
