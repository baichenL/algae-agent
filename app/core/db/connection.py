import sqlite3

from app.core.workspaces import DATABASE_PATH


DB_PATH = DATABASE_PATH


def connect(row_factory: bool = False):
    conn = sqlite3.connect(DB_PATH)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn
