"""Guarded read-only SQLite execution for validated analytical queries."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from receipt_intelligence.rag_sql.models import SqlExecutionResult, ValidatedSqlPlan
from receipt_intelligence.rag_sql.ports import AnalyticalQueryError
from receipt_intelligence.rag_sql.schema_catalog import (
    ALLOWED_ANALYTICS_OBJECTS,
    ALLOWED_SQL_FUNCTIONS,
    VIEW_BASE_TABLES,
    VIEW_COLUMNS,
)
from receipt_intelligence.storage.connection import SQLiteConnectionFactory


class SQLiteAnalyticalQueryRepository:
    """Execute allowlisted analytics SQL through a read-only SQLite connection."""

    def __init__(self, database_path: Path | str) -> None:
        self.connections = SQLiteConnectionFactory(database_path)

    def execute(
        self,
        plan: ValidatedSqlPlan,
        *,
        maximum_rows: int,
        timeout_seconds: float,
        progress_opcodes: int,
    ) -> SqlExecutionResult:
        started = time.perf_counter()
        deadline = started + timeout_seconds
        denied: list[str] = []
        try:
            connection = self.connections.connect_read_only(timeout_seconds=timeout_seconds)
        except FileNotFoundError as exc:
            raise AnalyticalQueryError(str(exc)) from exc

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
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.set_progress_handler(progress_handler, progress_opcodes)
            connection.set_authorizer(authorizer)
            cursor = connection.execute(plan.sql, dict(plan.parameters))
            columns = [description[0] for description in (cursor.description or [])]
            fetched = cursor.fetchmany(maximum_rows + 1)
            truncated = len(fetched) > maximum_rows
            rows = [dict(row) for row in fetched[:maximum_rows]]
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
                raise AnalyticalQueryError(
                    f"SQL execution exceeded {timeout_seconds:.2f} seconds{detail}."
                ) from exc
            raise AnalyticalQueryError(f"Read-only SQL execution failed: {exc}{detail}") from exc
        finally:
            connection.set_progress_handler(None, 0)
            connection.set_authorizer(None)
            connection.close()

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


__all__ = ["SQLiteAnalyticalQueryRepository"]
