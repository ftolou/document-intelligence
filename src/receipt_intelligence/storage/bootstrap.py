"""Explicit startup initialization for the receipt database schema."""

from __future__ import annotations

from pathlib import Path

from receipt_intelligence.storage.connection import SQLiteConnectionFactory
from receipt_intelligence.storage.migrations import MigrationRunner


def initialize_database(database_path: Path | str) -> list[int]:
    """Apply pending migrations once at an application or script entry point."""

    return MigrationRunner(SQLiteConnectionFactory(database_path)).migrate()


__all__ = ["initialize_database"]
