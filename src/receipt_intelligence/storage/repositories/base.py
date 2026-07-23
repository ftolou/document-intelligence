"""Common repository infrastructure."""

from __future__ import annotations

import sqlite3

from receipt_intelligence.storage.connection import SQLiteConnectionFactory


class BaseRepository:
    def __init__(self, connections: SQLiteConnectionFactory) -> None:
        self.connections = connections

    def connect(self) -> sqlite3.Connection:
        return self.connections.connect()


def fts_available(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SELECT COUNT(*) FROM receipt_item_fts").fetchone()
        return True
    except sqlite3.OperationalError:
        return False
