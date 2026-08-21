"""Explicit SQL-dialect contracts for generic RAG-SQL planning and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SqlDialectName = Literal["sqlite", "postgresql"]

_SQLITE_ALLOWED_FUNCTIONS = frozenset(
    {
        "abs",
        "avg",
        "cast",
        "coalesce",
        "count",
        "date",
        "datetime",
        "glob",
        "ifnull",
        "julianday",
        "like",
        "length",
        "lower",
        "ltrim",
        "max",
        "min",
        "nullif",
        "printf",
        "replace",
        "round",
        "rtrim",
        "strftime",
        "substr",
        "sum",
        "time",
        "total",
        "trim",
        "upper",
    }
)

# Keep the PostgreSQL profile intentionally conservative. Receipt/date filters already
# use typed columns and protected bind parameters, so dialect-specific date formatting
# functions are not required for the generic analytics contract.
_POSTGRESQL_ALLOWED_FUNCTIONS = frozenset(
    {
        "abs",
        "avg",
        "cast",
        "coalesce",
        "count",
        "length",
        "lower",
        "ltrim",
        "max",
        "min",
        "nullif",
        "replace",
        "round",
        "rtrim",
        "substr",
        "sum",
        "trim",
        "upper",
    }
)


@dataclass(frozen=True, slots=True)
class SqlDialectProfile:
    """Planner/validator policy for one supported SQL execution dialect."""

    name: SqlDialectName
    planner_label: str
    allowed_functions: frozenset[str]


SQLITE_DIALECT = SqlDialectProfile(
    name="sqlite",
    planner_label="SQLite",
    allowed_functions=_SQLITE_ALLOWED_FUNCTIONS,
)
POSTGRESQL_DIALECT = SqlDialectProfile(
    name="postgresql",
    planner_label="PostgreSQL",
    allowed_functions=_POSTGRESQL_ALLOWED_FUNCTIONS,
)

_DIALECTS: dict[str, SqlDialectProfile] = {
    SQLITE_DIALECT.name: SQLITE_DIALECT,
    POSTGRESQL_DIALECT.name: POSTGRESQL_DIALECT,
}


def get_sql_dialect_profile(name: str | SqlDialectName) -> SqlDialectProfile:
    """Resolve a supported dialect name to its immutable policy profile."""

    normalized = str(name or "").strip().casefold()
    try:
        return _DIALECTS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_DIALECTS))
        raise ValueError(
            f"Unsupported SQL dialect {name!r}; expected one of: {supported}."
        ) from exc


__all__ = [
    "POSTGRESQL_DIALECT",
    "SQLITE_DIALECT",
    "SqlDialectName",
    "SqlDialectProfile",
    "get_sql_dialect_profile",
]
