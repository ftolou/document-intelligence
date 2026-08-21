"""Conservative validation for LLM-generated read-only receipt analytics SQL.

The validator intentionally avoids a new parser dependency. It performs strict
lexical checks using the selected generic SQL-dialect profile. SQLite execution is
additionally paired with the authorizer in ``executor.py`` as a final local-runtime
object/function access boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from receipt_intelligence.rag_sql.filter_definitions import get_filter_definition
from receipt_intelligence.rag_sql.models import (
    JsonScalar,
    RagSqlPlanResult,
    ResolvedQueryFilter,
    ValidatedSqlPlan,
)
from receipt_intelligence.rag_sql.schema_catalog import ALLOWED_ANALYTICS_OBJECTS
from receipt_intelligence.rag_sql.sql_dialect import (
    SqlDialectProfile,
    get_sql_dialect_profile,
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
    sql_dialect: str = "sqlite"

    def __post_init__(self) -> None:
        if self.maximum_sql_length <= 0:
            raise ValueError("maximum_sql_length must be positive.")
        if self.maximum_rows <= 0 or self.maximum_rows > 1000:
            raise ValueError("maximum_rows must be between 1 and 1000.")
        profile = get_sql_dialect_profile(self.sql_dialect)
        object.__setattr__(self, "sql_dialect", profile.name)

    @property
    def dialect_profile(self) -> SqlDialectProfile:
        return get_sql_dialect_profile(self.sql_dialect)

    @property
    def allowed_functions(self) -> frozenset[str]:
        return self.dialect_profile.allowed_functions


class RagSqlValidator:
    def __init__(self, config: SqlValidatorConfig | None = None) -> None:
        self.config = config or SqlValidatorConfig()
        self.sql_dialect = self.config.dialect_profile

    def validate(
        self,
        plan: RagSqlPlanResult,
        *,
        protected_parameters: Mapping[str, JsonScalar] | None = None,
        resolved_filters: Sequence[ResolvedQueryFilter] | None = None,
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
        if scan.has_double_colon_cast:
            raise SqlValidationError(
                "Double-colon cast syntax is not allowed; use CAST(expression AS type) instead."
            )

        normalized = scan.masked.casefold().strip()
        first_token_match = re.search(r"[a-z_][a-z0-9_]*", normalized)
        first_token = first_token_match.group(0) if first_token_match else ""
        if first_token not in {"select", "with"}:
            raise SqlValidationError("SQL must begin with SELECT or WITH.")
        if re.match(r"\s*with\s+recursive\b", normalized):
            raise SqlValidationError("Recursive CTEs are not allowed.")

        tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", normalized))
        forbidden_keywords = sorted(
            keyword
            for keyword in _FORBIDDEN_KEYWORDS
            if re.search(rf"\b{re.escape(keyword)}\b(?!\s*\()", normalized)
        )
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

        _validate_resolved_filter_bindings(
            normalized,
            resolved_filters or [],
            protected_parameters,
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


def _validate_resolved_filter_bindings(
    normalized_sql: str,
    resolved_filters: Sequence[ResolvedQueryFilter],
    protected_parameters: Mapping[str, JsonScalar],
) -> None:
    for resolved_filter in resolved_filters:
        if resolved_filter.status != "resolved":
            continue
        parameter_names = sorted(
            name
            for name in protected_parameters
            if name.startswith(f"{resolved_filter.filter_id}_")
        )
        if not parameter_names:
            raise SqlValidationError(
                f"Resolved filter {resolved_filter.filter_id!r} has no protected parameters."
            )
        columns = get_filter_definition(resolved_filter.field).sql_columns
        if not _filter_binding_is_valid(
            normalized_sql,
            columns=columns,
            operator=resolved_filter.operator,
            parameter_names=parameter_names,
        ):
            rendered_columns = ", ".join(columns)
            raise SqlValidationError(
                f"Protected filter {resolved_filter.filter_id!r} must constrain "
                f"{rendered_columns} using operator {resolved_filter.operator!r}."
            )


def _filter_binding_is_valid(
    normalized_sql: str,
    *,
    columns: tuple[str, ...],
    operator: str,
    parameter_names: list[str],
) -> bool:
    placeholders = [rf":{re.escape(name)}\b" for name in parameter_names]

    if operator == "between":
        if len(placeholders) != 2:
            return False
        for allowed_column in columns:
            column = _qualified_column_pattern(allowed_column)
            direct_between = rf"{column}\s+between\s+{placeholders[0]}\s+and\s+{placeholders[1]}"
            comparison_pair = (
                rf"(?=.*{column}\s*>=\s*{placeholders[0]})"
                rf"(?=.*{column}\s*<=\s*{placeholders[1]})"
            )
            if re.search(direct_between, normalized_sql) or re.search(
                comparison_pair,
                normalized_sql,
                flags=re.DOTALL,
            ):
                return True
        return False

    comparison_operator = {
        "greater_than": ">",
        "greater_than_or_equal": ">=",
        "less_than": "<",
        "less_than_or_equal": "<=",
        "before": "<",
        "after": ">",
    }.get(operator)
    if comparison_operator is not None:
        if len(placeholders) != 1:
            return False
        escaped_operator = re.escape(comparison_operator)
        reverse_operator = {
            ">": "<",
            ">=": "<=",
            "<": ">",
            "<=": ">=",
        }[comparison_operator]
        for allowed_column in columns:
            column = _qualified_column_pattern(allowed_column)
            forward = rf"{column}\s*{escaped_operator}\s*{placeholders[0]}"
            reverse = rf"{placeholders[0]}\s*{re.escape(reverse_operator)}\s*{column}"
            if re.search(forward, normalized_sql) or re.search(reverse, normalized_sql):
                return True
        return False

    if operator not in {"matches", "equals", "contains", "in"}:
        return False

    for allowed_column in columns:
        column = _qualified_column_pattern(allowed_column)
        if len(placeholders) == 1:
            equality = (
                rf"(?:{column}\s*=\s*{placeholders[0]}|"
                rf"{placeholders[0]}\s*=\s*{column})"
            )
            if re.search(equality, normalized_sql):
                return True

        in_predicates = list(re.finditer(rf"{column}\s+in\s*\((?P<body>[^)]*)\)", normalized_sql))
        if any(
            all(re.search(placeholder, match.group("body")) for placeholder in placeholders)
            for match in in_predicates
        ):
            return True
    return False


def _qualified_column_pattern(column: str) -> str:
    return rf"(?:\b[a-z_][a-z0-9_]*\s*\.\s*)?\b{re.escape(column)}\b"


@dataclass(frozen=True)
class _SqlScan:
    masked: str
    named_parameters: list[str]
    statement_count: int
    has_comment: bool
    has_trailing_semicolon: bool
    has_positional_parameter: bool
    has_double_colon_cast: bool


def _scan_sql(sql: str) -> _SqlScan:
    masked: list[str] = []
    named_parameters: list[str] = []
    semicolon_positions: list[int] = []
    has_comment = False
    has_positional = False
    has_double_colon_cast = False
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
        if char == ":" and next_char == ":":
            has_double_colon_cast = True
            masked.extend([":", ":"])
            index += 2
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
        has_double_colon_cast=has_double_colon_cast,
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
