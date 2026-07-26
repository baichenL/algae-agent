from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import uuid
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from app.core.workspaces import (
    DEFAULT_CONTROL_DB_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_DOMAIN_DB_PATH,
)


DOMAIN_TABLES = ("algae_status", "experiments", "algae_audit", "domain_executions")


def inspect_legacy_workspace(legacy_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    source = Path(legacy_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    uri = f"file:{source.as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        quick_check = [
            str(row[0])
            for row in conn.execute("PRAGMA quick_check").fetchall()
        ]
        table_names = sorted(
            row[0]
            for row in conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        )
    return {
        "status": "passed" if quick_check == ["ok"] else "failed",
        "legacy_db_path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": digest.hexdigest().upper(),
        "quick_check": quick_check,
        "table_count": len(table_names),
        "domain_tables_present": [
            table for table in table_names if table in DOMAIN_TABLES
        ],
        "control_tables_present": [
            table for table in table_names if table not in DOMAIN_TABLES
        ],
    }


def _table_names(path: Path) -> list[str]:
    with closing(sqlite3.connect(path)) as conn:
        return sorted(
            row[0]
            for row in conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        )


def _classify_copy(path: Path, *, keep_domain: bool) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        tables = [
            row[0]
            for row in conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        ]
        for table in tables:
            is_domain = table in DOMAIN_TABLES
            if is_domain != keep_domain:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.commit()


def _table_digest(path: Path, table: str) -> dict[str, Any]:
    with closing(sqlite3.connect(path)) as conn:
        column_rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        columns = [row[1] for row in column_rows]
        if not columns:
            return {
                "count": 0,
                "primary_key_count": 0,
                "primary_keys_hash": None,
                "hash": None,
            }
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        rows = conn.execute(
            f'SELECT * FROM "{table}" ORDER BY {quoted_columns}'
        ).fetchall()
        primary_key_columns = [
            row[1]
            for row in sorted(column_rows, key=lambda item: int(item[5] or 0))
            if int(row[5] or 0) > 0
        ]
        column_index = {column: index for index, column in enumerate(columns)}
        primary_keys = [
            tuple(row[column_index[column]] for column in primary_key_columns)
            for row in rows
        ]
    payload = json.dumps(rows, ensure_ascii=False, default=str, separators=(",", ":"))
    primary_key_payload = json.dumps(
        primary_keys,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return {
        "count": len(rows),
        "primary_key_count": len(primary_keys) if primary_key_columns else 0,
        "primary_keys_hash": (
            hashlib.sha256(primary_key_payload.encode("utf-8")).hexdigest()
            if primary_key_columns
            else None
        ),
        "hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source)) as source_conn, closing(
        sqlite3.connect(destination)
    ) as target_conn:
        source_conn.backup(target_conn)


@contextmanager
def _workspace_write_lock(source: Path):
    lock_path = source.with_name(f"{source.name}.migration.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"workspace_migration_locked:{lock_path}") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def migrate_legacy_workspace(
    *,
    legacy_path: str = DEFAULT_DB_PATH,
    control_path: str = DEFAULT_CONTROL_DB_PATH,
    domain_path: str = DEFAULT_DOMAIN_DB_PATH,
    manifest_path: str = "data/workspace_storage_manifest.json",
    force: bool = False,
) -> dict[str, Any]:
    source = Path(legacy_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    preflight = inspect_legacy_workspace(str(source))
    if preflight["status"] != "passed":
        raise RuntimeError(
            "legacy_database_integrity_failed:"
            + ",".join(preflight["quick_check"])
        )
    control = Path(control_path).resolve()
    domain = Path(domain_path).resolve()
    if control == domain or source in {control, domain}:
        raise ValueError("storage_paths_must_be_distinct")
    if not force and (control.exists() or domain.exists()):
        raise FileExistsError("split_storage_already_exists")

    suffix = f".migrating-{uuid.uuid4().hex}"
    control_tmp = control.with_name(control.name + suffix)
    domain_tmp = domain.with_name(domain.name + suffix)
    try:
        with _workspace_write_lock(source):
            _backup(source, control_tmp)
            _backup(source, domain_tmp)
            _classify_copy(control_tmp, keep_domain=False)
            _classify_copy(domain_tmp, keep_domain=True)
            legacy_tables = _table_names(source)
            domain_tables = [
                table for table in legacy_tables if table in DOMAIN_TABLES
            ]
            control_tables = [
                table for table in legacy_tables if table not in DOMAIN_TABLES
            ]
            verification = {
                "domain": {
                    table: {
                        "legacy": _table_digest(source, table),
                        "split": _table_digest(domain_tmp, table),
                    }
                    for table in domain_tables
                },
                "control": {
                    table: {
                        "legacy": _table_digest(source, table),
                        "split": _table_digest(control_tmp, table),
                    }
                    for table in control_tables
                },
            }
            mismatches = [
                f"{group}:{table}"
                for group, tables in verification.items()
                for table, values in tables.items()
                if values["legacy"] != values["split"]
            ]
            if mismatches:
                raise RuntimeError(
                    f"storage_verification_failed:{','.join(mismatches)}"
                )
            control.parent.mkdir(parents=True, exist_ok=True)
            domain.parent.mkdir(parents=True, exist_ok=True)
            os.replace(control_tmp, control)
            os.replace(domain_tmp, domain)
            manifest = {
                "version": 1,
                "storage_mode": "split",
                "legacy_db_path": str(source),
                "control_db_path": str(control),
                "domain_db_path": str(domain),
                "legacy_preflight": preflight,
                "verification": verification,
            }
            manifest_file = Path(manifest_path).resolve()
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            manifest_tmp = manifest_file.with_name(manifest_file.name + suffix)
            manifest_tmp.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(manifest_tmp, manifest_file)
            # The legacy database remains a rollback artifact and is never
            # written after the manifest switch.
            try:
                source.chmod(stat.S_IREAD)
            except OSError:
                pass
            return manifest
    finally:
        for temporary in (control_tmp, domain_tmp):
            if temporary.exists():
                temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a legacy workspace SQLite database into split control/domain stores."
    )
    parser.add_argument("--legacy", default=DEFAULT_DB_PATH)
    parser.add_argument("--control", default=DEFAULT_CONTROL_DB_PATH)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN_DB_PATH)
    parser.add_argument(
        "--manifest",
        default="data/workspace_storage_manifest.json",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = inspect_legacy_workspace(args.legacy)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "passed":
            raise SystemExit(1)
        return
    result = migrate_legacy_workspace(
        legacy_path=args.legacy,
        control_path=args.control,
        domain_path=args.domain,
        manifest_path=args.manifest,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
