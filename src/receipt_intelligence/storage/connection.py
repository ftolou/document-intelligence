"""SQLite connection creation for the receipt intelligence store."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteConnectionFactory:
    """Create consistently configured SQLite connections.

    A factory keeps connection policy out of repositories and makes storage code
    straightforward to test with temporary database files.
    """

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
