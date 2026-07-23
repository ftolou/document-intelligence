"""Read-only SQLite execution with authorizer and timeout guards."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from receipt_intelligence.rag_sql.models import SqlExecutionResult, ValidatedSqlPlan
from receipt_intelligence.rag_sql.schema_catalog import (
    ALLOWED_ANALYTICS_OBJECTS,
    ALLOWED_SQL_FUNCTIONS,
    VIEW_BASE_TABLES,
    VIEW_COLUMNS,
)


class SqlExecutionError(RuntimeError):
    """Raised when validated SQL cannot be executed safely."""


@dataclass(frozen=True)
class ReadOnlySqlExecutorConfig:
    maximum_rows: int = 100
    timeout_seconds: float = 5.0
    progress_opcodes: int = 1000

    def __post_init__(self) -> None:
        if self.maximum_rows <= 0 or self.maximum_rows > 1000:
            raise ValueError("maximum_rows must be between 1 and 1000.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.progress_opcodes <= 0:
            raise ValueError("progress_opcodes must be positive.")


class ReadOnlySqlExecutor:
    def __init__(
        self,
        database_path: Path | str,
        config: ReadOnlySqlExecutorConfig | None = None,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.config = config or ReadOnlySqlExecutorConfig()

    def execute(self, plan: ValidatedSqlPlan) -> SqlExecutionResult:
        if not self.database_path.exists():
            raise SqlExecutionError(f"Receipt database does not exist: {self.database_path}")

        started = time.perf_counter()
        deadline = started + self.config.timeout_seconds
        connection = self._open_read_only()
        denied: list[str] = []

        def progress_handler() -> int:
            return 1 if time.perf_counter() > deadline else 0

        def authorizer(
            action: int,
            arg1: str | None,
            arg2: str | None,
            database_name: str | None,
            source: str | None,
        ) -> int:
            del database_name
            decision = self._authorize(action, arg1, arg2, source)
            if decision == sqlite3.SQLITE_DENY and len(denied) < 10:
                denied.append(f"action={action}, arg1={arg1!r}, arg2={arg2!r}, source={source!r}")
            return decision

        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.set_progress_handler(progress_handler, self.config.progress_opcodes)
            connection.set_authorizer(authorizer)
            cursor = connection.execute(plan.sql, dict(plan.parameters))
            columns = [description[0] for description in (cursor.description or [])]
            fetched = cursor.fetchmany(self.config.maximum_rows + 1)
            truncated = len(fetched) > self.config.maximum_rows
            rows = [dict(row) for row in fetched[: self.config.maximum_rows]]
            return SqlExecutionResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        except sqlite3.DatabaseError as exc:
            detail = f"; denied={denied}" if denied else ""
            if "interrupted" in str(exc).casefold():
                raise SqlExecutionError(
                    f"SQL execution exceeded {self.config.timeout_seconds:.2f} seconds{detail}."
                ) from exc
            raise SqlExecutionError(f"Read-only SQL execution failed: {exc}{detail}") from exc
        finally:
            connection.set_progress_handler(None, 0)
            connection.set_authorizer(None)
            connection.close()

    def _open_read_only(self) -> sqlite3.Connection:
        uri = f"{self.database_path.as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=self.config.timeout_seconds)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _authorize(
        action: int,
        arg1: str | None,
        arg2: str | None,
        source: str | None,
    ) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK

        recursive_action = getattr(sqlite3, "SQLITE_RECURSIVE", None)
        if recursive_action is not None and action == recursive_action:
            return sqlite3.SQLITE_OK

        if action == sqlite3.SQLITE_READ:
            object_name = str(arg1 or "").casefold()
            column_name = str(arg2 or "").casefold()
            source_name = str(source or "").casefold()

            if object_name in ALLOWED_ANALYTICS_OBJECTS:
                allowed_columns = VIEW_COLUMNS.get(object_name, frozenset())
                return (
                    sqlite3.SQLITE_OK
                    if not column_name or column_name in allowed_columns
                    else sqlite3.SQLITE_DENY
                )

            # SQLite expands views and reports reads from their base tables. The
            # source argument proves the read originated inside an allowlisted
            # view rather than from direct LLM SQL.
            if source_name in ALLOWED_ANALYTICS_OBJECTS:
                if object_name in VIEW_BASE_TABLES.get(source_name, frozenset()):
                    return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        if action == sqlite3.SQLITE_FUNCTION:
            function_name = str(arg2 or arg1 or "").casefold()
            return (
                sqlite3.SQLITE_OK if function_name in ALLOWED_SQL_FUNCTIONS else sqlite3.SQLITE_DENY
            )

        return sqlite3.SQLITE_DENY


__all__ = [
    "ReadOnlySqlExecutor",
    "ReadOnlySqlExecutorConfig",
    "SqlExecutionError",
]
