"""Small, explicit SQLite migration runner.

The runner is intentionally dependency-free. Migrations are idempotent so it can
adopt databases created by older versions of the application without data loss.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from receipt_intelligence.storage.connection import SQLiteConnectionFactory

LATEST_SCHEMA_VERSION = 6


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


class MigrationRunner:
    def __init__(self, connections: SQLiteConnectionFactory) -> None:
        self.connections = connections
        self.sql_directory = Path(__file__).with_name("sql")

    def migrate(self) -> list[int]:
        applied_now: list[int] = []
        with self.connections.connect() as connection:
            self._ensure_migration_table(connection)
            applied = {
                int(row["version"])
                for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
            }
            for migration in self._migrations():
                if migration.version in applied:
                    continue
                migration.apply(connection)
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) "
                    "VALUES (?, ?, datetime('now'))",
                    (migration.version, migration.name),
                )
                applied_now.append(migration.version)
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(LATEST_SCHEMA_VERSION),),
            )
            connection.commit()
        return applied_now

    def current_version(self) -> int:
        with self.connections.connect() as connection:
            self._ensure_migration_table(connection)
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
            ).fetchone()
            return int(row["version"] or 0)

    def _ensure_migration_table(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def _migrations(self) -> list[Migration]:
        return [
            Migration(1, "initial_schema", self._apply_initial_schema),
            Migration(2, "compatibility_columns_and_indexes", self._apply_compatibility),
            Migration(3, "receipt_item_fts", self._apply_fts),
            Migration(4, "rag_embedding_storage", self._apply_rag_embedding_storage),
            Migration(5, "rag_sql_analytics_views", self._apply_rag_sql_analytics_views),
            Migration(6, "reviewed_product_semantics", self._apply_reviewed_product_semantics),
        ]

    def _apply_sql(self, connection: sqlite3.Connection, filename: str) -> None:
        sql = (self.sql_directory / filename).read_text(encoding="utf-8")
        connection.executescript(sql)

    def _apply_initial_schema(self, connection: sqlite3.Connection) -> None:
        self._apply_sql(connection, "001_initial_schema.sql")

    def _apply_compatibility(self, connection: sqlite3.Connection) -> None:
        # Older databases used CREATE TABLE followed by ad-hoc ALTER statements.
        # Adopt them by adding only missing columns, then create current indexes.
        self._add_missing_columns(
            connection,
            "receipt_items",
            {
                "parser_item_type": "TEXT",
                "original_price": "REAL",
                "discount_amount": "REAL",
                "tax_code": "TEXT",
            },
        )
        self._add_missing_columns(
            connection,
            "receipts",
            {
                "file_sha256": "TEXT",
                "content_fingerprint": "TEXT",
                "duplicate_status": "TEXT",
                "duplicate_of_receipt_id": "INTEGER",
                "duplicate_score": "REAL",
            },
        )
        self._apply_sql(connection, "002_indexes.sql")

    def _apply_fts(self, connection: sqlite3.Connection) -> None:
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS receipt_item_fts USING fts5(
                    item_id UNINDEXED,
                    receipt_id UNINDEXED,
                    merchant,
                    receipt_date UNINDEXED,
                    raw_name,
                    normalized_name,
                    category,
                    embedding_text
                )
                """
            )
            connection.execute(
                """
                INSERT INTO receipt_item_fts(
                    item_id, receipt_id, merchant, receipt_date, raw_name,
                    normalized_name, category, embedding_text
                )
                SELECT i.id, i.receipt_id,
                       trim(COALESCE(r.merchant_name, '') || ' ' ||
                            COALESCE(r.merchant_normalized, '')),
                       r.receipt_date, i.raw_name, i.normalized_name,
                       i.category, i.embedding_text
                FROM receipt_items i
                JOIN receipts r ON r.id = i.receipt_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM receipt_item_fts f WHERE f.item_id = i.id
                )
                """
            )
        except sqlite3.OperationalError:
            # The query layer has a deterministic lexical fallback when the
            # embedded SQLite library does not include FTS5.
            pass

    def _apply_rag_embedding_storage(self, connection: sqlite3.Connection) -> None:
        self._apply_sql(connection, "004_rag_embeddings.sql")

    def _apply_rag_sql_analytics_views(self, connection: sqlite3.Connection) -> None:
        self._apply_sql(connection, "005_rag_sql_analytics_views.sql")

    def _apply_reviewed_product_semantics(self, connection: sqlite3.Connection) -> None:
        self._add_missing_columns(
            connection,
            "receipt_items",
            {
                "category_reason": "TEXT",
                "semantic_description": "TEXT",
            },
        )
        rows = connection.execute(
            "SELECT id, raw_json, category_reason, semantic_description FROM receipt_items"
        ).fetchall()
        for row in rows:
            category_reason = row["category_reason"]
            semantic_description = row["semantic_description"]
            if category_reason not in (None, "") and semantic_description not in (None, ""):
                continue
            try:
                payload = json.loads(row["raw_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            if category_reason in (None, ""):
                category_reason = payload.get("category_reason")
            if semantic_description in (None, ""):
                semantic_description = payload.get("semantic_description")
            connection.execute(
                "UPDATE receipt_items SET category_reason=?, semantic_description=? WHERE id=?",
                (
                    str(category_reason).strip() if category_reason not in (None, "") else None,
                    str(semantic_description).strip()
                    if semantic_description not in (None, "")
                    else None,
                    int(row["id"]),
                ),
            )
        self._apply_sql(connection, "006_reviewed_product_semantics.sql")

    @staticmethod
    def _add_missing_columns(
        connection: sqlite3.Connection,
        table: str,
        columns: dict[str, str],
    ) -> None:
        existing = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, sql_type in columns.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
