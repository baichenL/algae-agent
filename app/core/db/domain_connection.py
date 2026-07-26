from __future__ import annotations

import os
import sqlite3
from typing import Any

from app.core.workspaces import current_workspace


def connect_domain_readonly(path: Any) -> sqlite3.Connection:
    resolved = os.fspath(path)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def connect_domain_writer(path: Any) -> sqlite3.Connection:
    trusted = os.getenv("TRUSTED_WORKER_PROCESS", "false").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    test_mode = os.getenv("ALGAE_AUTH_MODE", "required").casefold() == "test"
    if current_workspace().storage_mode == "split" and not (trusted or test_mode):
        raise PermissionError("domain_fact_writer_requires_trusted_worker")
    return sqlite3.connect(path)
