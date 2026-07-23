"""Conservative validation for LLM-generated SQLite SELECT statements.

The validator intentionally avoids a new parser dependency. It performs strict
lexical checks and is paired with the SQLite authorizer in ``executor.py``.
The authorizer is the final object/function access boundary at execution time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from receipt_intelligence.rag_sql.models import RagSqlPlanResult, ValidatedSqlPlan
from receipt_intelligence.rag_sql.schema_catalog import (
    ALLOWED_ANALYTICS_OBJECTS,
    ALLOWED_SQL_FUNCTIONS,
)

_FORBIDDEN_KEYWORDS = frozenset(
    {
        "alter",
        "analyze",
        "attach",
        "begin",
        "commit",
        "create",
        "delete",
        "detach",
        "drop",
        "end",
        "explain",
        "insert",
        "load_extension",
        "pragma",
        "reindex",
        "release",
        "replace",
        "rollback",
        "savepoint",
        "transaction",
        "update",
        "vacuum",
    }
)

_FORBIDDEN_OBJECTS = frozenset(
    {
        "receipts",
        "receipt_items",
        "rag_item_embeddings",
        "rag_index_state",
        "review_queue",
        "duplicate_candidates",
        "product_aliases",
        "schema_meta",
        "schema_migrations",
        "sqlite_master",
        "sqlite_schema",
        "sqlite_temp_master",
        "receipt_item_fts",
    }
)

_NON_FUNCTION_TOKENS = frozenset(
    {
        "as",
        "case",
        "distinct",
        "filter",
        "in",
        "not",
        "on",
        "over",
        "select",
        "when",
    }
)


class SqlValidationError(ValueError):
    """Raised when SQL violates the RAG-SQL execution contract."""


@dataclass(frozen=True)
class SqlValidatorConfig:
    maximum_sql_length: int = 20000
    maximum_rows: int = 100
    allowed_objects: frozenset[str] = ALLOWED_ANALYTICS_OBJECTS
    allowed_functions: frozenset[str] = ALLOWED_SQL_FUNCTIONS

    def __post_init__(self) -> None:
        if self.maximum_sql_length <= 0:
            raise ValueError("maximum_sql_length must be positive.")
        if self.maximum_rows <= 0 or self.maximum_rows > 1000:
            raise ValueError("maximum_rows must be between 1 and 1000.")


class RagSqlValidator:
    def __init__(self, config: SqlValidatorConfig | None = None) -> None:
        self.config = config or SqlValidatorConfig()

    def validate(
        self,
        plan: RagSqlPlanResult,
        *,
        protected_parameters: Mapping[str, int] | None = None,
    ) -> ValidatedSqlPlan:
        if plan.status != "ready" or not plan.sql or not plan.result_shape:
            raise SqlValidationError("Only a ready SQL plan can be validated.")

        sql = plan.sql.strip()
        if not sql:
            raise SqlValidationError("SQL must not be empty.")
        if len(sql) > self.config.maximum_sql_length:
            raise SqlValidationError(f"SQL exceeds {self.config.maximum_sql_length} characters.")

        scan = _scan_sql(sql)
        if scan.has_comment:
            raise SqlValidationError("SQL comments are not allowed.")
        if scan.statement_count != 1:
            raise SqlValidationError("Exactly one SQL statement is allowed.")
        if scan.has_trailing_semicolon:
            raise SqlValidationError("A trailing semicolon is not allowed.")

        normalized = scan.masked.casefold().strip()
        first_token_match = re.search(r"[a-z_][a-z0-9_]*", normalized)
        first_token = first_token_match.group(0) if first_token_match else ""
        if first_token not in {"select", "with"}:
            raise SqlValidationError("SQL must begin with SELECT or WITH.")
        if re.match(r"\s*with\s+recursive\b", normalized):
            raise SqlValidationError("Recursive CTEs are not allowed.")

        tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", normalized))
        forbidden_keywords = sorted(tokens & _FORBIDDEN_KEYWORDS)
        if forbidden_keywords:
            raise SqlValidationError(f"Forbidden SQL keyword(s): {', '.join(forbidden_keywords)}.")
        forbidden_objects = sorted(tokens & _FORBIDDEN_OBJECTS)
        if forbidden_objects:
            raise SqlValidationError(
                f"Direct access to storage object(s) is forbidden: {', '.join(forbidden_objects)}."
            )

        if (plan.result_entity or "").casefold() in {"brand", "product_brand"}:
            merchant_tokens = {"merchant", "merchant_name"} & tokens
            if merchant_tokens:
                raise SqlValidationError(
                    "Merchant fields identify the seller and cannot be used to answer a product-brand query."
                )
        if re.search(r"\bmerchant(?:_name)?\s+as\s+(?:product_)?brand\b", normalized):
            raise SqlValidationError(
                "merchant AS brand is forbidden because merchant identifies the seller."
            )

        referenced_objects = _extract_referenced_objects(normalized)
        unknown_objects = sorted(
            object_name
            for object_name in referenced_objects
            if object_name not in self.config.allowed_objects
            and object_name not in _extract_cte_names(normalized)
        )
        if unknown_objects:
            raise SqlValidationError(
                f"SQL references non-allowlisted object(s): {', '.join(unknown_objects)}."
            )
        if not (set(referenced_objects) & set(self.config.allowed_objects)):
            raise SqlValidationError("SQL must read from an allowlisted analytics view.")

        referenced_functions = _extract_functions(normalized)
        unknown_functions = sorted(
            function_name
            for function_name in referenced_functions
            if function_name not in self.config.allowed_functions
            and function_name not in _NON_FUNCTION_TOKENS
        )
        if unknown_functions:
            raise SqlValidationError(
                f"SQL uses non-allowlisted function(s): {', '.join(unknown_functions)}."
            )

        if scan.has_positional_parameter:
            raise SqlValidationError("Only named :parameter placeholders are allowed.")
        placeholder_names = sorted(set(scan.named_parameters))
        parameter_names = sorted(plan.parameters)
        if placeholder_names != parameter_names:
            missing_values = sorted(set(placeholder_names) - set(parameter_names))
            unused_values = sorted(set(parameter_names) - set(placeholder_names))
            raise SqlValidationError(
                "SQL placeholders and parameters must match exactly. "
                f"missing_values={missing_values}, unused_values={unused_values}."
            )

        protected_parameters = protected_parameters or {}
        for name, expected_value in protected_parameters.items():
            if name not in plan.parameters:
                raise SqlValidationError(f"Protected parameter {name!r} is missing.")
            if plan.parameters[name] != expected_value:
                raise SqlValidationError(f"Protected parameter {name!r} was modified.")
            if name not in placeholder_names:
                raise SqlValidationError(
                    f"Protected parameter {name!r} is not used in the SQL statement."
                )

        if plan.result_shape == "row":
            limit = _literal_limit(normalized)
            if limit != 1:
                raise SqlValidationError("row result_shape requires literal LIMIT 1.")
        elif plan.result_shape in {"rows", "grouped_rows"}:
            limit = _literal_limit(normalized)
            if limit is None:
                raise SqlValidationError(
                    f"{plan.result_shape} result_shape requires a literal LIMIT."
                )
            if limit <= 0 or limit > self.config.maximum_rows:
                raise SqlValidationError(f"LIMIT must be between 1 and {self.config.maximum_rows}.")

        return ValidatedSqlPlan(
            sql=sql,
            parameters=plan.parameters,
            result_shape=plan.result_shape,
            result_entity=plan.result_entity or "result",
            display_columns=plan.display_columns,
            answer_instruction=plan.answer_instruction or "Present the query result.",
            referenced_objects=sorted(set(referenced_objects)),
            referenced_functions=sorted(set(referenced_functions)),
            placeholder_names=placeholder_names,
        )


@dataclass(frozen=True)
class _SqlScan:
    masked: str
    named_parameters: list[str]
    statement_count: int
    has_comment: bool
    has_trailing_semicolon: bool
    has_positional_parameter: bool


def _scan_sql(sql: str) -> _SqlScan:
    masked: list[str] = []
    named_parameters: list[str] = []
    semicolon_positions: list[int] = []
    has_comment = False
    has_positional = False
    index = 0
    length = len(sql)
    quote: str | None = None

    while index < length:
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < length else ""

        if quote is not None:
            masked.append(" ")
            if quote == "[":
                if char == "]":
                    quote = None
            elif char == quote:
                if next_char == quote and quote in {"'", '"'}:
                    masked.append(" ")
                    index += 1
                else:
                    quote = None
            index += 1
            continue

        if char in {"'", '"', "`", "["}:
            quote = char
            masked.append(" ")
            index += 1
            continue
        if char == "-" and next_char == "-":
            has_comment = True
            masked.extend([" ", " "])
            index += 2
            while index < length and sql[index] not in "\r\n":
                masked.append(" ")
                index += 1
            continue
        if char == "/" and next_char == "*":
            has_comment = True
            masked.extend([" ", " "])
            index += 2
            while index < length:
                if sql[index] == "*" and index + 1 < length and sql[index + 1] == "/":
                    masked.extend([" ", " "])
                    index += 2
                    break
                masked.append(" ")
                index += 1
            continue
        if char == ";":
            semicolon_positions.append(index)
            masked.append(";")
            index += 1
            continue
        if char == "?" or char in {"@", "$"}:
            has_positional = True
            masked.append(char)
            index += 1
            continue
        if char == ":":
            match = re.match(r":([A-Za-z][A-Za-z0-9_]*)", sql[index:])
            if match:
                name = match.group(1)
                named_parameters.append(name)
                token = match.group(0)
                masked.extend(token)
                index += len(token)
                continue
        masked.append(char)
        index += 1

    if quote is not None:
        raise SqlValidationError("SQL contains an unterminated quoted literal or identifier.")

    stripped = sql.rstrip()
    has_trailing_semicolon = stripped.endswith(";")
    if semicolon_positions:
        statement_count = len(semicolon_positions) + (0 if has_trailing_semicolon else 1)
    else:
        statement_count = 1

    return _SqlScan(
        masked="".join(masked),
        named_parameters=named_parameters,
        statement_count=statement_count,
        has_comment=has_comment,
        has_trailing_semicolon=has_trailing_semicolon,
        has_positional_parameter=has_positional,
    )


def _extract_cte_names(normalized_sql: str) -> set[str]:
    names: set[str] = set()
    if not re.match(r"\s*with\b", normalized_sql):
        return names
    for match in re.finditer(r"(?:\bwith\b|,)\s*([a-z_][a-z0-9_]*)\s+as\s*\(", normalized_sql):
        names.add(match.group(1))
    return names


def _extract_referenced_objects(normalized_sql: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\b", normalized_sql)
    ]


def _extract_functions(normalized_sql: str) -> list[str]:
    return [match.group(1) for match in re.finditer(r"\b([a-z_][a-z0-9_]*)\s*\(", normalized_sql)]


def _literal_limit(normalized_sql: str) -> int | None:
    matches = list(re.finditer(r"\blimit\s+([0-9]+)\b", normalized_sql))
    if not matches:
        return None
    return int(matches[-1].group(1))


__all__ = [
    "RagSqlValidator",
    "SqlValidationError",
    "SqlValidatorConfig",
]
