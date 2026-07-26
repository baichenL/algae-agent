import sqlite3

import os

from app.core.workspaces import CONTROL_DATABASE_PATH, DOMAIN_DATABASE_PATH


DB_PATH = CONTROL_DATABASE_PATH
DOMAIN_DB_PATH = DOMAIN_DATABASE_PATH


def connect(row_factory: bool = False):
    conn = sqlite3.connect(DB_PATH)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def connect_domain_readonly(row_factory: bool = False):
    path = os.fspath(DOMAIN_DB_PATH)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def connect_domain_writer(row_factory: bool = False):
    if os.getenv("TRUSTED_WORKER_PROCESS", "false").casefold() not in {
        "1", "true", "yes", "on",
    } and os.getenv("ALGAE_AUTH_MODE", "required").casefold() != "test":
        raise PermissionError("domain_fact_writer_requires_trusted_worker")
    conn = sqlite3.connect(DOMAIN_DB_PATH)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn
