"""SQLite connection creation for the receipt intelligence store."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class ClosingSQLiteConnection(sqlite3.Connection):
    """Commit or roll back a context-managed transaction, then release the file handle."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


class SQLiteConnectionFactory:
    """Create consistently configured SQLite connections.

    A factory keeps connection policy out of repositories and makes storage code
    straightforward to test with temporary database files.
    """

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, factory=ClosingSQLiteConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def connect_read_only(self, *, timeout_seconds: float = 5.0) -> sqlite3.Connection:
        """Open an existing database using SQLite's read-only URI mode."""

        database_path = self.database_path.resolve()
        if not database_path.exists():
            raise FileNotFoundError(f"Receipt database does not exist: {database_path}")
        uri = f"{database_path.as_uri()}?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=timeout_seconds,
            factory=ClosingSQLiteConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection
