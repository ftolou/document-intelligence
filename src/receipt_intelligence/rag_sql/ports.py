"""Persistence contract for guarded analytical SQL execution."""

from __future__ import annotations

from typing import Protocol

from receipt_intelligence.rag_sql.models import SqlExecutionResult, ValidatedSqlPlan


class AnalyticalQueryError(RuntimeError):
    """Raised when validated analytical SQL cannot be executed safely."""


class AnalyticalQueryRepository(Protocol):
    """Execute a validated read-only query through a storage adapter."""

    def execute(
        self,
        plan: ValidatedSqlPlan,
        *,
        maximum_rows: int,
        timeout_seconds: float,
        progress_opcodes: int,
    ) -> SqlExecutionResult: ...


__all__ = ["AnalyticalQueryError", "AnalyticalQueryRepository"]
