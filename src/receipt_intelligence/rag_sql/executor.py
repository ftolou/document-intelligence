"""Application-level execution of validated analytical SQL."""

from __future__ import annotations

from dataclasses import dataclass

from receipt_intelligence.rag_sql.models import SqlExecutionResult, ValidatedSqlPlan
from receipt_intelligence.rag_sql.ports import (
    AnalyticalQueryError,
    AnalyticalQueryRepository,
)

SqlExecutionError = AnalyticalQueryError


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
    """Delegate validated SQL to an injected read-only persistence adapter."""

    def __init__(
        self,
        repository: AnalyticalQueryRepository,
        config: ReadOnlySqlExecutorConfig | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or ReadOnlySqlExecutorConfig()

    def execute(self, plan: ValidatedSqlPlan) -> SqlExecutionResult:
        return self.repository.execute(
            plan,
            maximum_rows=self.config.maximum_rows,
            timeout_seconds=self.config.timeout_seconds,
            progress_opcodes=self.config.progress_opcodes,
        )


__all__ = [
    "ReadOnlySqlExecutor",
    "ReadOnlySqlExecutorConfig",
    "SqlExecutionError",
]
